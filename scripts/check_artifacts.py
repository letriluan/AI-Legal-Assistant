"""Offline check of committed examples/config; never load pickle or model weights."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
examples = json.loads((root / "src/legal_assistant/configs/preview_examples.json").read_text(encoding="utf-8"))
assert examples and len({row["question"] for row in examples}) == len(examples)
for example in examples:
    assert example["answer"] and example["sources"]
    assert example["answer_en"] and example["question_en"] and example["translation_notice"]
    assert all(source["article"] for source in example["sources"])
print(f"Validated {len(examples)} archived preview examples; no LLM evaluation performed.")
