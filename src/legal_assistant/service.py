"""Explicit saved-output preview and opt-in full RAG service."""
import json
import threading
import unicodedata

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
                if example:
                    return {**example, "original_input": question, "mode": "preview", "latency_ms": 0}
                return {"question": question, "answer": "This preview only replays the listed submission examples. "
                        "Choose an example, or enable full RAG mode to ask a new question.",
                        "sources": [], "mode": "preview", "latency_ms": 0}
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
