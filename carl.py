#!/usr/bin/env python3
"""CARL -- Chat Assisted RL

A conversational interface that guides users through designing
and building reinforcement learning systems, step by step.

Usage:
    python carl.py

Environment variables:
    AWS_REGION                   (optional) Bedrock region, defaults to us-west-2
    CARL_MODEL                   (optional) Main workflow model, defaults to
                                 Opus 5 with a Sonnet 5 fallback
    CARL_REPORT_MODEL            (optional) Report writing model, defaults to Sonnet 5
    CARL_PERSONA_UPDATE_MODEL    (optional) Post-session persona rewrite model,
                                 defaults to Sonnet 5
    CARL_TRILLIAN_MODEL          (optional) Suggestion model, defaults to Haiku 4.5
"""

import os
import sys
from pathlib import Path


def _load_dotenv(carl_dir: str):
    """Load .env file from the CARL directory if it exists."""
    env_path = Path(carl_dir) / ".env"
    if not env_path.is_file():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CARL -- Chat Assisted RL")
    parser.add_argument(
        "--write-report", action="store_true",
        help="Generate a LaTeX+PDF academic report after all steps complete",
    )
    parser.add_argument(
        "--no-persona-update", action="store_true",
        help="Skip the Trillian persona update at session end (use when running multiple instances)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--revisit",
        help="Open an existing CARL session directory directly in Step 10 revisit mode",
    )
    mode.add_argument(
        "--resume",
        help="Resume an existing CARL session from its last unfinished step",
    )
    args = parser.parse_args()

    carl_dir = str(Path(__file__).resolve().parent)

    _load_dotenv(carl_dir)

    try:
        import boto3  # noqa: F401
    except ImportError:
        print("Error: boto3 package not installed.")
        print("  pip install boto3")
        sys.exit(1)

    from chat import CARLChat
    from resume import ResumeError

    try:
        chat = CARLChat(
            carl_dir,
            write_report=args.write_report,
            no_persona_update=args.no_persona_update,
            revisit_dir=args.revisit,
            resume_dir=args.resume,
        )
    except ResumeError as e:
        print(f"Cannot resume: {e}")
        sys.exit(2)
    chat.run()


if __name__ == "__main__":
    main()
