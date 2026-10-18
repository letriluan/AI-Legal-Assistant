"""Explicit saved-output preview and opt-in full RAG service."""
import json
import threading
import unicodedata

from .archive import find_answer
from .settings import CONFIG_DIR, mode


def normalize(question):
    return " ".join(unicodedata.normalize("NFC", question).casefold().split())


class Service:
    def __init__(self):
        self.lock = threading.Lock()
        self.engine = None
        self.error = False
        self.examples = json.loads((CONFIG_DIR / "preview_examples.json").read_text(encoding="utf-8"))

    def ask(self, question):
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("Another request is running")
        try:
            example = next((row for row in self.examples if normalize(question) in {
                normalize(row["question"]), normalize(row.get("question_en", ""))}), None)
            if mode() == "preview":
                archived = find_answer(question)
                if archived:
                    sources = []
                    for reference in archived.get("relevant_articles", []):
                        parts = reference.split("|")
                        sources.append({"code": parts[0], "document": parts[1] if len(parts)>1 else "",
                                        "article": parts[2] if len(parts)>2 else "",
                                        "excerpt": "Archived citation only; source text is not included in this preview."})
                    return {**archived, "sources": sources, **(example or {}),
                            "archive_id": archived["id"], "matched": True, "answer_language": "vi",
                            "original_input": question, "mode": "preview", "latency_ms": 0,
                            "review_note": (example or {}).get("review_note", "Archived model output; not verified legal advice or ground truth.")}
                return {"question": question, "answer": "This preview only replays archived submission answers. "
                        "No matching question was found. Search the question library, or run full RAG locally for new questions.",
                        "matched": False, "sources": [], "mode": "preview", "latency_ms": 0}
            if self.engine is None:
                from .runtime import RagEngine
                self.engine = RagEngine()
            self.error = False
            # Only curated example translations map back to Vietnamese. No general
            # English translation capability is claimed for full inference.
            result = self.engine.ask(example["question"] if example else question)
            return {**result, "original_input": question}
        except Exception:
            self.error = True
            raise
        finally:
            self.lock.release()
