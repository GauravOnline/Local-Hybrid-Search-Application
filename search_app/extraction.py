"""Read text from the supported local file formats."""

import hashlib
import io
import os
import subprocess
from pathlib import Path

import fitz
from docx import Document
from PIL import Image, ImageOps

from .config import CHUNK_OVERLAP, CHUNK_WORDS, SUPPORTED_TYPES


def file_hash(path: Path) -> str:
    """Calculate a SHA-256 hash without loading the entire file into memory."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def ocr_image(image: Image.Image) -> str:
    """Run the installed Tesseract executable against a printed-text image."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    command = os.environ.get("TESSERACT_CMD", "tesseract")
    try:
        result = subprocess.run(
            [command, "stdin", "stdout", "-l", "eng"],
            input=buffer.getvalue(), capture_output=True, check=True, timeout=90,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Tesseract is not installed or not on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.decode(errors="replace").strip()) from exc
    return result.stdout.decode("utf-8").strip()


def extract_text(path: Path) -> str:
    """Extract document text, using OCR for images and image-only PDF pages."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix == ".txt":
        return path.read_text(encoding="utf-8-sig", errors="replace").strip()
    if suffix == ".docx":
        document = Document(path)
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            paragraphs.extend(" | ".join(cell.text for cell in row.cells) for row in table.rows)
        return "\n".join(paragraphs).strip()
    if suffix == ".pdf":
        pages = []
        with fitz.open(path) as document:
            for page in document:
                text = page.get_text().strip()
                if not text:
                    # A scanned page has no searchable text layer.
                    pixmap = page.get_pixmap(dpi=200, alpha=False)
                    with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
                        text = ocr_image(image)
                pages.append(text)
        return "\n\n".join(pages).strip()
    with Image.open(path) as image:
        return ocr_image(image)


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split words into overlapping chunks to preserve context at boundaries."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Chunk size must be positive and overlap smaller than size.")
    words = text.split()
    chunks = []
    for start in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks
