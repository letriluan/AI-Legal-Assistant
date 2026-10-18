import json
import sqlite3
import unicodedata
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from legal_assistant.app import app
from legal_assistant.archive import connect, count_answers, find_answer, find_questions

ROOT = Path(__file__).resolve().parents[1]


def test_every_original_answer_and_reference_is_preserved():
    rows = json.loads((ROOT / "archive/submission_answers.json").read_text(encoding="utf-8"))
    assert count_answers() == len(rows) == 2000
    for row in rows:
        assert find_answer(row["question"]) == row
    assert find_answer("  " + unicodedata.normalize("NFD", rows[0]["question"].upper()) + "  ") == rows[0]
    assert find_answer(rows[0]["question"] + " invented additional question") is None


def test_search_is_literal_and_database_is_read_only():
    assert len(find_questions("", 20)) == 20
    assert find_questions("bao hiem") == find_questions("bảo hiểm")
    assert find_questions("%' OR 1=1 --") == []
    connection = connect()
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM answers")
    finally:
        connection.close()


def test_archive_api_and_non_curated_answer(monkeypatch):
    monkeypatch.setenv("APP_MODE", "preview")
    with TestClient(app) as client:
        rows = client.get("/archive/questions?limit=20").json()["questions"]
        answer = client.post("/ask", json={"question": rows[-1]["question"]}).json()
        assert answer["matched"] and answer["archive_id"] == rows[-1]["id"]
        assert "answer_en" not in answer
        assert client.get("/archive/questions?limit=100").status_code == 422
        assert client.get("/examples").json()["archive_count"] == 2000


def test_missing_database_fails_readiness(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_MODE", "preview")
    monkeypatch.setenv("PREVIEW_DB_PATH", str(tmp_path / "missing.sqlite"))
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503
