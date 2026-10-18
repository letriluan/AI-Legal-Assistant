"""Read-only lookup of archived submission answers, never semantic answer matching."""
import hashlib
import json
import os
import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path

from .settings import ROOT


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def search_text(text):
    return "".join(char for char in unicodedata.normalize("NFD", normalize(text).replace("đ", "d"))
                   if unicodedata.category(char) != "Mn")


def archive_path():
    return Path(os.getenv("PREVIEW_DB_PATH", str(ROOT / "cache/preview.sqlite"))).resolve()


def connect():
    path = archive_path().resolve(strict=True)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def build_archive(source, output, examples, manifest):
    source, output = Path(source), Path(output)
    metadata = json.loads(Path(manifest).read_text(encoding="utf-8"))
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise ValueError("Archive source checksum mismatch")
    rows = json.loads(raw)
    if len(rows) != metadata["records"]:
        raise ValueError("Archive record count mismatch")
    translations = json.loads(Path(examples).read_text(encoding="utf-8"))
    originals = {normalize(row["question"]): row for row in rows}
    if len(originals) != len(rows):
        raise ValueError("Duplicate normalized archive questions")
    for example in translations:
        original = originals[normalize(example["question"])]
        if original["answer"] != example["answer"]:
            raise ValueError("Translation no longer matches the archived Vietnamese answer")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build to a separate file; an interrupted build never replaces a working database.
    temporary = output.with_suffix(".building.sqlite")
    try:
        with closing(sqlite3.connect(temporary)) as connection:
            connection.executescript("""
                DROP TABLE IF EXISTS aliases;
                DROP TABLE IF EXISTS answers;
                CREATE TABLE answers (id INTEGER PRIMARY KEY, normalized TEXT UNIQUE NOT NULL,
                    question TEXT NOT NULL, search TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE aliases (normalized TEXT PRIMARY KEY, answer_id INTEGER NOT NULL
                    REFERENCES answers(id));
            """)
            connection.executemany("INSERT INTO answers VALUES (?,?,?,?,?)", [
                (row["id"], normalize(row["question"]), row["question"], search_text(row["question"]),
                 json.dumps(row, ensure_ascii=False)) for row in rows])
            connection.executemany("INSERT INTO aliases VALUES (?,?)", [
                (normalize(example["question_en"]), originals[normalize(example["question"])]["id"])
                for example in translations])
            connection.commit()
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(rows)


def count_answers():
    with closing(connect()) as connection:
        return connection.execute("SELECT COUNT(*) FROM answers").fetchone()[0]


def find_answer(question):
    key = normalize(question)
    with closing(connect()) as connection:
        row = connection.execute("""SELECT payload FROM answers WHERE normalized=?
            UNION ALL SELECT a.payload FROM aliases x JOIN answers a ON a.id=x.answer_id
            WHERE x.normalized=? LIMIT 1""", (key, key)).fetchone()
    return json.loads(row[0]) if row else None


def find_questions(query, limit=12):
    needle = search_text(query).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    with closing(connect()) as connection:
        rows = connection.execute("""SELECT id, question FROM answers
            WHERE search LIKE ? ESCAPE '\\' ORDER BY id LIMIT ?""", ("%" + needle + "%", limit)).fetchall()
    return [{"id": row[0], "question": row[1]} for row in rows]
