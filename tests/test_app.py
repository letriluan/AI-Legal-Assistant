import importlib
import sys

import pytest
from fastapi.testclient import TestClient

from legal_assistant import app as module


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("APP_MODE", "preview")
    with TestClient(module.app) as client:
        yield client


def test_import_does_not_load_model_runtime():
    importlib.reload(module)
    assert "sentence_transformers" not in sys.modules
    assert module.service.engine is None


def test_preview_is_explicit_and_serves_static_assets(client):
    assert client.get("/health").json()["mode"] == "preview"
    assert client.get("/ready").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/ui/app.js").status_code == 200
    examples = client.get("/examples").json()
    assert len(examples["questions"]) == 6
    response = client.post("/ask", json={"question": examples["questions"][0]})
    data = response.json()
    assert data["mode"] == "preview" and data["sources"] and data["request_id"]


def test_unknown_question_is_not_fabricated(client):
    data = client.post("/ask", json={"question": "This is a new unrelated question"}).json()
    assert "only replays" in data["answer"] and not data["sources"]


@pytest.mark.parametrize("question", ["", "  ", "x"*2001])
def test_request_validation(client, question):
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_errors_do_not_expose_configuration(client, monkeypatch):
    def fail(question):
        raise RuntimeError("secret-internal-path")
    monkeypatch.setattr(module.service, "ask", fail)
    response = client.post("/ask", json={"question": "A new question"})
    assert response.status_code == 503 and "secret-internal-path" not in response.text


def test_concurrent_request_returns_retry_after(client):
    module.service.lock.acquire()
    try:
        response = client.post("/ask", json={"question": "A new question"})
        assert response.status_code == 429 and response.headers["Retry-After"] == "5"
    finally:
        module.service.lock.release()


def test_rag_readiness_does_not_claim_loaded_models(client, monkeypatch):
    monkeypatch.setenv("APP_MODE", "rag")
    assert client.get("/ready").status_code == 503


def test_checksum_mismatch_is_rejected_before_pickle_loading(tmp_path):
    import hashlib
    import json

    from legal_assistant.runtime import ArtifactError, verify_artifacts
    name = "bm25_dual_with_docs_phapdien64_web_canonical.pkl"
    (tmp_path / name).write_bytes(b"untrusted pickle payload")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({name: {"sha256": hashlib.sha256(b"expected").hexdigest()}}))
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        verify_artifacts(tmp_path, manifest)


@pytest.mark.parametrize("index", range(6))
def test_curated_english_and_vietnamese_questions_replay_the_same_original(client, index):
    examples = client.get("/examples").json()["examples"]
    example = examples[index]
    assert example["question_en"] and example["topic_en"]
    english = client.post("/ask", json={"question": example["question_en"]}).json()
    vietnamese = client.post("/ask", json={"question": example["question"]}).json()
    assert english["mode"] == "preview"
    assert english["answer"] == vietnamese["answer"]
    assert english["answer_en"] == vietnamese["answer_en"]
    assert english["sources"] == vietnamese["sources"]
    assert english["answer_language"] == "vi"
    assert "not an official" in english["translation_notice"]
    assert english["original_input"] == example["question_en"]
    assert all(source["code"] and source["document_en"] for source in english["sources"])


def test_live_rag_maps_curated_english_question_without_replaying_archived_translation(client, monkeypatch):
    example = module.service.examples[0]
    received = []
    class Engine:
        def ask(self, question):
            received.append(question)
            return {"answer": "Câu trả lời mới.", "sources": [], "mode": "rag", "latency_ms": 1}
    monkeypatch.setenv("APP_MODE", "rag")
    monkeypatch.setattr(module.service, "engine", Engine())
    response = client.post("/ask", json={"question": example["question_en"]}).json()
    assert received == [example["question"]]
    assert response["answer"] == "Câu trả lời mới."
    assert "answer_en" not in response


def test_archived_accounting_reference_mismatch_is_disclosed(client):
    example = module.service.examples[5]
    response = client.post("/ask", json={"question": example["question_en"]}).json()
    assert "58/2021/TT-BTC" in response["answer_en"]
    assert "133/2016/TT-BTC" in response["review_note"]
    assert "mismatch" in response["review_note"]
