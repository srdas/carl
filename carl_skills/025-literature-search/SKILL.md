---
name: 025-literature-search
description: Search academic literature for relevant papers and extract RL-formulation details
metadata:
  sources: []
---

# Step 2.5: Literature Search

Search for academic papers related to the user's problem to inform the RL formulation.

## Purpose

Find published work that can help define states, actions, rewards, and parameter
ranges for the RL system being designed. This is NOT a general literature review --
focus exclusively on information that feeds into subsequent design steps.

## How to Use the Tool

You have one tool available: `search_literature(query, limit)`

- Construct queries from the problem description keywords (Step 1) combined with
  relevant terms like "reinforcement learning", "optimal control", "dynamic programming",
  or domain-specific terms from any references provided in Step 2.
- Start with a broad query, then refine based on what comes back.
- Limit to 5-10 results per query. Quality over quantity.

## Workflow

1. Review the problem definition (Step 1) and any references (Step 2).
2. Construct 1-2 targeted search queries and call `search_literature`.
3. Present results as a numbered list with title, authors, year, citations, and
   a one-line description of what the paper covers.
4. Ask the user which papers look relevant (they may select by number).
5. If the user wants, refine with a different query. Otherwise, proceed.
6. Summarize ONLY the following from selected papers:
   - **State representations** used (what the agent observes)
   - **Action spaces** (continuous/discrete, bounds, constraints)
   - **Reward functions** (shaping, terminal, discount conventions)
   - **Parameter ranges** (typical values, units, domain bounds)
   - **Analytical solutions or baselines** that can serve as benchmarks or
     pre-training targets
7. Do NOT produce a general summary of contributions, novelty, or methodology.

## If the User Wants to Skip

The user can type `/next` to skip this step entirely. Do not insist on searching
if the user indicates they don't need a literature search.

## Checklist

- [ ] At least one search query executed
- [ ] User reviewed results and selected relevant papers (or declined)
- [ ] RL-formulation-relevant findings extracted from selected papers
- [ ] Summary focuses on states, actions, rewards, parameters, and baselines
