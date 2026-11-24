"""Shared application settings."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SEARCH_DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = Path(os.environ.get("SEARCH_MODEL_DIR", PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"))
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSION = 384
CHUNK_WORDS = 160
CHUNK_OVERLAP = 30
RRF_K = 60
SUPPORTED_TYPES = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}
