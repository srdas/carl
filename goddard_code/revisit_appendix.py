#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
#
# This file is part of CARL.
#
# CARL is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3, as published by the
# Free Software Foundation.
#
# CARL is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for details.
#
# You should have received a copy of the GNU General Public License along
# with CARL. If not, see <https://www.gnu.org/licenses/>.

"""Publish a report-style Step 10 revisit appendix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Write experiments/<run>/revisit_section.tex but do not publish to results/.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Update LaTeX only; skip pdflatex even if available.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    carl_root = next(
        (p for p in [script_dir, *script_dir.parents] if (p / "revisit_report.py").is_file()),
        script_dir.parents[2],
    )
    if str(carl_root) not in sys.path:
        sys.path.insert(0, str(carl_root))

    from revisit_report import write_revisit_appendix

    written = write_revisit_appendix(
        args.run_name,
        args.run_dir,
        script_dir,
        publish=not args.local_only,
        compile_pdf=not args.no_pdf,
    )
    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
