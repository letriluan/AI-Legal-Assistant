"""Build the preview SQLite database from the committed archived submission JSON."""
import argparse
from pathlib import Path

from legal_assistant.archive import archive_path, build_archive
from legal_assistant.settings import CONFIG_DIR

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default=str(archive_path()))
args = parser.parse_args()
count = build_archive(root / "archive/submission_answers.json", args.output,
                      CONFIG_DIR / "preview_examples.json", root / "archive/manifest.json")
print(f"Built read-only preview source: {count} archived answers.")
