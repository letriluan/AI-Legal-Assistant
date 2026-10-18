"""Explicit opt-in batch runner for the original full-model workflow."""
import argparse
import json
from pathlib import Path

from legal_assistant.runtime import RagEngine

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--questions", required=True)
group = parser.add_mutually_exclusive_group()
group.add_argument("--limit", type=int, default=1)
group.add_argument("--all", action="store_true")
parser.add_argument("--resume-dir")
args = parser.parse_args()
if not args.all and args.limit < 1:
    parser.error("--limit must be positive")
questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
engine = RagEngine()
engine.core.run_experiment(questions, engine.config,
                           limit=None if args.all else args.limit, resume_dir=args.resume_dir)
