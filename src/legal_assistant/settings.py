"""Runtime configuration; imports do not load models or read the corpus."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[2]


def mode():
    value = os.getenv("APP_MODE", "preview")
    if value not in {"preview", "rag"}:
        raise ValueError("APP_MODE must be preview or rag")
    return value


def data_dir():
    return Path(os.getenv("LEGAL_DATA_DIR", "data/full")).resolve()

CONFIG_DIR = Path(__file__).parent / "configs"
