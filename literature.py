"""Academic literature search across arXiv, OpenAlex and Semantic Scholar.

Semantic Scholar alone is not dependable here: the keyless endpoint shares one
heavily-contended rate-limit pool and answers 429 for most requests, which the
old single-backend implementation reported as "No papers found" — indistinguishable
from a genuinely empty index. Sessions then burned a dozen reformulated queries
chasing a transport failure.

Backends are therefore tried in order and their results merged until *limit*
papers are collected:

    arXiv      — no key, no practical throttle, abstract + PDF on every hit;
                 the primary venue for RL / quantitative-finance preprints.
    OpenAlex   — no key, ~100k requests/day, peer-reviewed coverage and
                 citation counts that arXiv lacks.
    Semantic Scholar — best relevance ranking, but only worth trying first
                 when SEMANTIC_SCHOLAR_API_KEY is set; keyless it goes last.

Transport failures are reported to the caller separately from empty results so
the model can tell "backend down" from "nothing published on this".
"""

import io
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from bedrock_converse import converse_text

# ── Configuration ─────────────────────────────────────────────────────

TIMEOUT = 20  # seconds
MAX_RETRIES = 2
RETRY_DELAY = 3  # seconds (exponential: 3s, 6s)

_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
# OpenAlex asks for a contact address to route requests to the faster,
# more reliable "polite pool".
_CONTACT = os.environ.get("OPENALEX_CONTACT_EMAIL", "carl-rl-assistant@example.com")

_PAGE_SIZE = 20         # over-fetch per page to compensate for filtering
_MAX_PAGES = 2          # max Semantic Scholar pages (avoids hammering)

# Hard ceiling on HTTP requests for ONE search_papers() call, counting every
# backend, relaxation round and retry. Without it the worst case is ~27
# requests (3 backends x several rounds x 3 attempts); with it a search is
# bounded in both wall-clock and load no matter how the backends behave.
MAX_REQUESTS_PER_SEARCH = 8

# arXiv asks for no more than one request every 3 seconds.
_MIN_REQUEST_INTERVAL = {"export.arxiv.org": 3.0}
_last_request_at: dict[str, float] = {}

# Query relaxation: an AND of every term often matches nothing, so terms are
# dropped shortest-first (shortest ≈ least distinctive) and the results
# accumulated across rounds.
_MAX_TERMS = 6
_MAX_RELAX_ROUNDS = 4

_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with",
    "using", "use", "via", "under", "from", "its", "their", "how", "what",
    "problem", "problems", "approach", "approaches", "based", "study",
    "paper", "papers", "method", "methods", "new", "novel",
}


# ── Search ────────────────────────────────────────────────────────────

class _RequestBudget:
    """Shared HTTP request allowance for one search.

    Backends check it before every call and stop cleanly when it runs out,
    returning whatever they have gathered so far. Running out is not an error —
    it is the cap doing its job — so it is never reported as a failure.
    """

    def __init__(self, allowance: int):
        self.remaining = max(1, allowance)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def take(self) -> bool:
        if self.exhausted:
            return False
        self.remaining -= 1
        return True


def search_papers(query: str, limit: int = 10) -> list[dict]:
    """Search the configured backends for papers matching *query*.

    Only papers with a non-empty abstract are returned (the abstract feeds
    the LLM summary); a PDF URL is preferred but not required, and results
    that have one are ordered first.

    Returns a list of dicts with keys:
        title, authors, year, abstract, citation_count, paper_id, url,
        pdf_url, source
    Returns [] if every backend comes back empty or fails.
    """
    papers, _ = search_papers_with_status(query, limit)
    return papers


def search_papers_with_status(
    query: str, limit: int = 10,
) -> tuple[list[dict], list[str]]:
    """Like :func:`search_papers`, but also returns backend error strings.

    The second element is a list of human-readable failures (one per backend
    that errored). It lets callers distinguish an empty index from an
    unreachable API instead of telling the model to rephrase a fine query.
    """
    errors: list[str] = []
    pools: list[list[dict]] = []
    # No single backend may claim every slot: arXiv alone would fill the list
    # with preprints and no citation counts, so each gets a quota first and
    # the leftovers only fill whatever is still empty.
    quota = max(1, (limit * 2 + 2) // 3)
    budget = _RequestBudget(MAX_REQUESTS_PER_SEARCH)

    for name, fetch in _backends():
        if sum(min(len(p), quota) for p in pools) >= limit or budget.exhausted:
            break
        try:
            pools.append(fetch(query, limit, budget))
        except Exception as e:  # network, parse, or shape surprise
            errors.append(f"{name}: {e}")

    collected: list[dict] = []
    index: dict[str, dict] = {}

    def absorb(papers: list[dict]) -> None:
        for paper in papers:
            if len(collected) >= limit:
                return
            if not paper.get("abstract"):
                continue
            keys = _dedupe_keys(paper)
            kept = next((index[k] for k in keys if k in index), None)
            if kept is not None:
                # Same work from two backends — keep the richer record: the
                # preprint's PDF plus the published version's citation count.
                if paper.get("citation_count", 0) > kept.get("citation_count", 0):
                    kept["citation_count"] = paper["citation_count"]
                if not kept.get("pdf_url") and paper.get("pdf_url"):
                    kept["pdf_url"] = paper["pdf_url"]
                if not kept.get("doi") and paper.get("doi"):
                    kept["doi"] = paper["doi"]
                index.update({k: kept for k in keys})
                continue
            index.update({k: paper for k in keys})
            collected.append(paper)

    for pool in pools:
        absorb(pool[:quota])
    for pool in pools:  # overflow: fill any slots the quotas left empty
        absorb(pool)

    # Prefer entries with a downloadable PDF (the summariser reads the first
    # pages) while otherwise preserving backend relevance order.
    collected.sort(key=lambda p: not p.get("pdf_url"))
    return collected[:limit], errors


def _backends() -> list[tuple[str, object]]:
    """Backends in the order they should be tried."""
    order = [
        ("arXiv", _search_arxiv),
        ("OpenAlex", _search_openalex),
        ("Semantic Scholar", _search_semantic_scholar),
    ]
    if _API_KEY:
        # A key buys a private rate limit, so S2's ranking leads.
        order.insert(0, order.pop())
    return order


def _dedupe_keys(paper: dict) -> list[str]:
    """Every identity a paper can be recognised by.

    Both title and DOI are returned, not one in preference to the other:
    arXiv exposes no DOI while OpenAlex does, so keying on DOI alone would let
    the preprint and the published record through as two separate papers.
    """
    keys = []
    title = re.sub(r"[^a-z0-9]+", "", (paper.get("title") or "").lower())
    if title:
        keys.append(f"title:{title}")
    doi = (paper.get("doi") or "").lower().strip()
    if doi:
        keys.append(f"doi:{doi}")
    return keys


def _query_terms(query: str) -> list[str]:
    """Distinctive search terms, longest-biased, in query order."""
    words = re.findall(r"[A-Za-z0-9\-']+", query)
    terms, seen = [], set()
    for w in words:
        low = w.lower()
        if low in _STOPWORDS or len(w) <= 2 or low in seen:
            continue
        seen.add(low)
        terms.append(w)
    if len(terms) > _MAX_TERMS:
        keep = set(sorted(terms, key=len, reverse=True)[:_MAX_TERMS])
        terms = [t for t in terms if t in keep]
    return terms


# ── Backend: arXiv ────────────────────────────────────────────────────

_ATOM = "{http://www.w3.org/2005/Atom}"


def _search_arxiv(query: str, limit: int, budget: _RequestBudget) -> list[dict]:
    """Query the arXiv Atom API, relaxing the term set until enough hits."""
    terms = _query_terms(query)
    if not terms:
        return []

    out: list[dict] = []
    seen: set[str] = set()
    for _ in range(_MAX_RELAX_ROUNDS):
        search_query = " AND ".join(f'all:"{t}"' for t in terms)
        for paper in _arxiv_page(search_query, _PAGE_SIZE, budget):
            keys = _dedupe_keys(paper)
            if not any(k in seen for k in keys):
                seen.update(keys)
                out.append(paper)
        if len(out) >= limit or len(terms) <= 2 or budget.exhausted:
            break
        terms.remove(min(terms, key=len))
    return out


def _arxiv_page(
    search_query: str, page_size: int, budget: _RequestBudget,
) -> list[dict]:
    resp = _request(
        "https://export.arxiv.org/api/query",
        {
            "search_query": search_query,
            "max_results": page_size,
            "sortBy": "relevance",
        },
        budget,
        parse="text",
    )
    if not resp:
        return []

    papers = []
    for entry in ET.fromstring(resp).findall(f"{_ATOM}entry"):
        arxiv_url = _text(entry.find(f"{_ATOM}id"))
        pdf_url = ""
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
        published = _text(entry.find(f"{_ATOM}published"))
        papers.append({
            "title": _clean(_text(entry.find(f"{_ATOM}title"))),
            "authors": [
                _clean(_text(a.find(f"{_ATOM}name")))
                for a in entry.findall(f"{_ATOM}author")
            ],
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "abstract": _clean(_text(entry.find(f"{_ATOM}summary"))),
            # arXiv exposes no citation counts.
            "citation_count": 0,
            "paper_id": arxiv_url.rsplit("/", 1)[-1],
            "url": arxiv_url,
            "pdf_url": pdf_url or (arxiv_url.replace("/abs/", "/pdf/")
                                   if "/abs/" in arxiv_url else ""),
            "doi": "",
            "source": "arXiv",
        })
    return papers


def _text(node) -> str:
    return (node.text or "") if node is not None else ""


def _clean(text: str) -> str:
    return " ".join(text.split())


# ── Backend: OpenAlex ─────────────────────────────────────────────────

def _search_openalex(
    query: str, limit: int, budget: _RequestBudget,
) -> list[dict]:
    """Query OpenAlex, narrow filter first then a fuzzy fallback."""
    terms = _query_terms(query)
    attempts = []
    if terms:
        attempts.append({
            "filter": "title_and_abstract.search:" + " ".join(terms),
            "sort": "relevance_score:desc",
        })
        if len(terms) > 3:
            narrow = sorted(terms, key=len, reverse=True)[:3]
            attempts.append({
                "filter": "title_and_abstract.search:" + " ".join(narrow),
                "sort": "relevance_score:desc",
            })
    attempts.append({"search": query})

    for params in attempts:
        if budget.exhausted:
            break
        params = {**params, "per_page": _PAGE_SIZE, "mailto": _CONTACT}
        data = _request("https://api.openalex.org/works", params, budget)
        papers = [_openalex_paper(w) for w in (data or {}).get("results", [])]
        papers = [p for p in papers if p["abstract"]]
        if papers:
            return papers[:max(limit, 1) * 2]
    return []


def _openalex_paper(work: dict) -> dict:
    best = work.get("best_oa_location") or work.get("primary_location") or {}
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    return {
        "title": _clean(work.get("title") or work.get("display_name") or ""),
        "authors": [
            (a.get("author") or {}).get("display_name", "")
            for a in (work.get("authorships") or [])
        ],
        "year": work.get("publication_year"),
        "abstract": _openalex_abstract(work),
        "citation_count": work.get("cited_by_count", 0),
        "paper_id": (work.get("id") or "").rsplit("/", 1)[-1],
        "url": work.get("doi") or work.get("id") or "",
        "pdf_url": best.get("pdf_url") or "",
        "doi": doi,
        "source": "OpenAlex",
    }


def _openalex_abstract(work: dict) -> str:
    """Rebuild the abstract from OpenAlex's inverted index."""
    inverted = work.get("abstract_inverted_index")
    if not isinstance(inverted, dict):
        return _clean(work.get("abstract") or "")
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for i in idxs or []:
            positions.append((i, word))
    positions.sort()
    return _clean(" ".join(word for _, word in positions))


# ── Backend: Semantic Scholar ─────────────────────────────────────────

def _search_semantic_scholar(
    query: str, limit: int, budget: _RequestBudget,
) -> list[dict]:
    headers = {"x-api-key": _API_KEY} if _API_KEY else {}
    out: list[dict] = []
    for page in range(_MAX_PAGES):
        if budget.exhausted:
            break
        data = _request(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            {
                "query": query,
                "limit": _PAGE_SIZE,
                "offset": page * _PAGE_SIZE,
                "fields": "title,abstract,authors,year,citationCount,"
                          "externalIds,openAccessPdf",
            },
            budget,
            headers=headers,
        )
        items = (data or {}).get("data") or []
        if not items:
            break
        for item in items:
            pdf = (item.get("openAccessPdf") or {}).get("url")
            ids = item.get("externalIds") or {}
            out.append({
                "title": _clean(item.get("title", "")),
                "authors": [
                    a.get("name", "") for a in (item.get("authors") or [])
                ],
                "year": item.get("year"),
                "abstract": _clean(item.get("abstract") or ""),
                "citation_count": item.get("citationCount", 0),
                "paper_id": item.get("paperId", ""),
                "url": f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
                "pdf_url": pdf or "",
                "doi": (ids.get("DOI") or "").lower(),
                "source": "Semantic Scholar",
            })
        if len(out) >= limit:
            break
    return out


# ── HTTP ──────────────────────────────────────────────────────────────

def _throttle(url: str) -> None:
    """Respect per-host minimum request spacing."""
    host = url.split("/")[2]
    interval = _MIN_REQUEST_INTERVAL.get(host)
    if not interval:
        return
    elapsed = time.monotonic() - _last_request_at.get(host, 0.0)
    if elapsed < interval:
        time.sleep(interval - elapsed)


class LiteratureBackendError(RuntimeError):
    """A backend could not be reached (network error, rate limit, 5xx)."""


def _request(url, params, budget, headers=None, parse="json"):
    """GET with throttling, 429-aware retry and a shared request budget.

    Returns parsed JSON, or raw text when ``parse="text"``, or None once
    *budget* is spent (a cap, not a failure — the caller keeps what it has).
    Raises :class:`LiteratureBackendError` if the request genuinely failed, so
    a transport error is never mistaken for an empty result set.
    """
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        # Retries count against the budget too — a rate-limited host is
        # exactly the case where retrying must not run away.
        if not budget.take():
            if last_error:
                break
            return None
        _throttle(url)
        try:
            resp = requests.get(
                url, params=params, headers=headers or {}, timeout=TIMEOUT,
            )
            _last_request_at[url.split("/")[2]] = time.monotonic()
            if resp.status_code == 429:
                last_error = "429 rate-limited"
                if attempt < MAX_RETRIES:
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        wait = float(retry_after) if retry_after else 0
                    except ValueError:
                        wait = 0
                    time.sleep(max(wait, RETRY_DELAY * (attempt + 1)))
                    continue
                break
            resp.raise_for_status()
            return resp.text if parse == "text" else resp.json()
        except requests.RequestException as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
    host = url.split("/")[2]
    print(f"  [literature: {host} unavailable — {last_error}]")
    raise LiteratureBackendError(f"{host} unavailable — {last_error}")


# ── Formatting ────────────────────────────────────────────────────────

def format_results_for_tool(
    papers: list[dict], errors: list[str] | None = None,
) -> str:
    """Format search results as compact text for a tool_result block.

    *errors* are backend transport failures. They are reported verbatim so the
    model does not read a rate limit as evidence that the literature is thin
    and rephrase the same query a dozen times.
    """
    errors = errors or []
    if not papers:
        if errors:
            return (
                "Search backends failed — this is a transport problem, NOT an "
                "empty literature. Do not rephrase the query in response to "
                "this; either retry the same query later or proceed without "
                "the search.\nFailures: " + "; ".join(errors)
            )
        return "No papers found. Try a different query."

    lines = [f"Found {len(papers)} papers:\n"]
    if errors:
        lines.append(
            f"(partial results — some backends failed: {'; '.join(errors)})\n"
        )
    for i, p in enumerate(papers, 1):
        authors = p["authors"][:3]
        author_str = ", ".join(authors)
        if len(p["authors"]) > 3:
            author_str += f" (+{len(p['authors']) - 3} more)"

        abstract = p["abstract"]
        if len(abstract) > 200:
            abstract = abstract[:200] + "..."

        cite = p.get("citation_count") or 0
        source = p.get("source", "")
        origin = f"{source} (no citation data)" if source == "arXiv" else source
        lines.append(
            f"[{i}] {p['title']}\n"
            f"    Authors: {author_str} ({p['year'] or '?'})\n"
            f"    Citations: {cite}"
            + (f"    Source: {origin}\n" if origin else "\n")
            + f"    Abstract: {abstract}\n"
        )
    return "\n".join(lines)


# ── PDF reading ───────────────────────────────────────────────────────

_MAX_MD_PAPERS = 10  # max papers written to the markdown summary file


def _fetch_pdf_text(pdf_url: str, max_pages: int = 3) -> str:
    """Download a PDF and extract text from the first *max_pages* pages.

    Returns the extracted text or "" on any failure.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        resp = requests.get(
            pdf_url, timeout=30,
            headers={"User-Agent": "CARL/1.0"},
            allow_redirects=True,
        )
        if "pdf" not in resp.headers.get("content-type", ""):
            return ""
        reader = PdfReader(io.BytesIO(resp.content))
        parts = []
        for page in reader.pages[:max_pages]:
            text = page.extract_text()
            if text:
                parts.append(text)
        return "\n".join(parts)
    except Exception:
        return ""


def _summarize_paper(title: str, text: str, bedrock_client, model_id: str) -> str:
    """Call an LLM to produce a ~50-word summary of a paper.

    *text* is the first-three-pages PDF extract (preferred) or the abstract
    (fallback).  Returns the summary string or "" on failure.
    """
    if not text:
        return ""
    # Cap input to keep the call cheap
    text = text[:6000]
    try:
        return converse_text(
            bedrock_client, model_id,
            "Summarize in EXACTLY 40-50 words. One paragraph, no headings, "
            "no bullet points. Cover: RL algorithm used, state/action/reward "
            "design, and main result. Be terse.",
            [{"role": "user", "content": f"Title: {title}\n\n{text}"}],
            max_tokens=100, temperature=0.2,
        ).strip()
    except Exception:
        return ""


# ── Persistence ───────────────────────────────────────────────────────

def save_literature_results(
    session_dir,
    papers: list[dict],
    bedrock_client=None,
    summary_model: str = "",
    overwrite: bool = False,
) -> None:
    """Save search results to the session's references/ directory.

    If *bedrock_client* and *summary_model* are provided, each paper's PDF
    is fetched (first 2 pages), summarized via the LLM, and the summary is
    included in the markdown file.  The markdown file is capped at
    ``_MAX_MD_PAPERS`` entries.

    If *overwrite* is True, existing literature files are replaced so the
    current call defines the canonical shortlist for this run.
    """
    ref_dir = Path(session_dir) / "references"
    ref_dir.mkdir(exist_ok=True)

    # JSON (full data — no cap)
    json_path = ref_dir / "literature_search.json"
    existing = []
    if json_path.exists() and not overwrite:
        existing = json.loads(json_path.read_text())
    existing.extend(papers)
    json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    # Markdown summary (capped at _MAX_MD_PAPERS total)
    md_path = ref_dir / "literature_search.md"
    md_text = ""
    if md_path.exists() and not overwrite:
        md_text = md_path.read_text(encoding="utf-8")
    existing_count = sum(1 for line in md_text.splitlines() if line.startswith("- **"))
    remaining = max(0, _MAX_MD_PAPERS - existing_count)
    to_write = papers[:remaining]
    if not to_write:
        return

    mode = "w" if overwrite else "a"
    with open(md_path, mode, encoding="utf-8") as f:
        if mode == "a" and md_text and not md_text.endswith("\n"):
            f.write("\n")
        for p in to_write:
            authors = ", ".join(p["authors"][:3])
            summary = ""
            if bedrock_client and summary_model:
                source_text = ""
                if p.get("pdf_url"):
                    source_text = _fetch_pdf_text(p["pdf_url"])
                if not source_text and p.get("abstract"):
                    source_text = p["abstract"]
                summary = _summarize_paper(
                    p["title"], source_text, bedrock_client, summary_model,
                )
            f.write(f"- **{p['title']}** ({p['year'] or '?'}) — {authors}\n")
            if summary:
                f.write(f"  > {summary}\n")
