"""Unit tests and integrated retrieval checks."""

import hashlib
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import faiss
import fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scripts.evaluate import metrics, load_labels
from scripts.generate_dataset import write_file, generate
from search_app.config import MODEL_DIR
from search_app.extraction import chunk_text, extract_text, file_hash
from search_app.main import create_app
from search_app.service import SearchService, fts_query, reciprocal_rank_fusion


def test_hash_and_chunks(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"hello")
    assert file_hash(path) == hashlib.sha256(b"hello").hexdigest()
    assert chunk_text("one two three four five", 3, 1) == ["one two three", "three four five"]
    assert chunk_text("") == []
    with pytest.raises(ValueError):
        chunk_text("hello", 3, 3)


def test_rank_fusion():
    ranked = reciprocal_rank_fusion([[1, 2, 3], [2, 4, 1]], k=60)
    assert list(ranked) == [2, 1, 4, 3]
    assert ranked[2] == pytest.approx(1 / 62 + 1 / 61)
    assert reciprocal_rank_fusion([[], []]) == {}
    assert list(reciprocal_rank_fusion([[7, 7]])) == [7]
    assert fts_query('test "OR" (word)') == '"test" OR "OR" OR "word"'


@pytest.mark.parametrize("extension", [".txt", ".pdf", ".docx"])
def test_document_types(tmp_path, extension):
    path = tmp_path / ("example" + extension)
    text = "Unique reference ALPHA-752"
    write_file(path, text)
    assert extract_text(path) == text


def character_accuracy(expected, observed):
    """Levenshtein distance normalized by the original transcription length."""
    expected = " ".join(expected.split())
    observed = " ".join(observed.split())
    previous = list(range(len(observed) + 1))
    for index, char in enumerate(expected, 1):
        current = [index]
        for position, other in enumerate(observed, 1):
            current.append(min(current[-1] + 1, previous[position] + 1,
                               previous[position - 1] + (char != other)))
        previous = current
    return max(0, 1 - previous[-1] / len(expected))


@pytest.mark.skipif(not shutil.which(os.environ.get("TESSERACT_CMD", "tesseract")), reason="Tesseract not installed")
@pytest.mark.parametrize("extension", [".png", ".jpg", ".pdf"])
def test_ocr_and_search(tmp_path, service, extension):
    corpus = tmp_path / "source"
    corpus.mkdir()
    text = "LUMEN-204\nConference room reservation"
    path = corpus / ("scan" + extension)
    if extension == ".pdf":
        image_path = tmp_path / "image.png"
        write_file(image_path, text)
        with fitz.open() as document:
            page = document.new_page(width=600, height=175)
            page.insert_image(page.rect, filename=str(image_path))
            document.save(path)
    else:
        write_file(path, text)
    observed = extract_text(path)
    accuracy = character_accuracy(text, observed)
    assert accuracy >= 0.90
    assert service.index_folder(corpus)["errors"] == []
    assert service.search("LUMEN-204", "keyword")["results"][0]["name"] == path.name


def test_incremental_and_persistence(tmp_path, service):
    corpus = tmp_path / "source"
    corpus.mkdir()
    alpha = corpus / "alpha.txt"
    alpha.write_text("obsolete token")
    assert service.index_folder(corpus)["added"] == 1
    calls = service.encoder.calls
    assert service.index_folder(corpus)["unchanged"] == 1
    assert service.encoder.calls == calls
    alpha.write_text("replacement text")
    extra = corpus / "extra.txt"
    extra.write_text("fresh document")
    report = service.index_folder(corpus)
    assert (report["updated"], report["added"]) == (1, 1)
    assert not service.search("obsolete", "keyword")["results"]
    renamed = corpus / "renamed.txt"
    extra.rename(renamed)
    alpha.unlink()
    report = service.index_folder(corpus)
    assert (report["added"], report["removed"]) == (1, 2)
    assert service.status()["files"] == service.status()["vectors"] == 1
    assert service.search("fresh", "hybrid")["results"][0]["path"] == str(renamed)
    assert faiss.read_index(str(service.data_dir / "vectors.faiss")).ntotal == 1
    reopened = SearchService(service.data_dir, encoder=service.encoder)
    try:
        assert reopened.search("fresh", "semantic")["results"][0]["name"] == "renamed.txt"
        assert reopened.status()["chunks"] == 1
    finally:
        reopened.close()


def test_filter_deduplication_and_source(tmp_path, service):
    folder = tmp_path / "source"
    sub = folder / "nested"
    sub.mkdir(parents=True)
    document = sub / "report.txt"
    document.write_text("apple orchard " * 400)
    write_file(folder / "plan.pdf", "apple orchard schedule")
    assert service.index_folder(folder)["errors"] == []
    assert service.status()["chunks"] > 2
    for mode in ("keyword", "semantic", "hybrid"):
        results = service.search("apple orchard", mode, extension=".txt", folder=str(sub))["results"]
        assert len(results) == 1
        assert service.source(results[0]["id"]) == document
    assert len(service.search("apple", "semantic")["results"]) == 2
    assert service.index_folder(sub)["unchanged"] == 1
    assert service.status()["files"] == 2


def test_failed_replacement_has_no_stale_results(tmp_path, service):
    corpus = tmp_path / "source"
    corpus.mkdir()
    path = corpus / "document.pdf"
    write_file(path, "unique test document")
    service.index_folder(corpus)
    path.write_bytes(b"invalid PDF")
    report = service.index_folder(corpus)
    assert len(report["errors"]) == 1
    assert service.status()["files"] == service.status()["vectors"] == 0
    assert service.search("unique", "keyword")["results"] == []
    with pytest.raises(ValueError):
        service.index_folder(corpus / "missing")


def test_api(tmp_path, service):
    corpus = tmp_path / "source"
    corpus.mkdir()
    file = corpus / "note.txt"
    file.write_text("test content")
    with TestClient(create_app(service)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200
        assert client.post("/api/index", json={"folder": str(corpus)}).json()["added"] == 1
        result = client.post("/api/search", json={"query": "test", "mode": "hybrid"})
        assert result.status_code == 200
        file_id = result.json()["results"][0]["id"]
        if os.name == "nt":
            with patch("search_app.main.os.startfile") as opener:
                assert client.post(f"/api/files/{file_id}/open", json={}).status_code == 200
                opener.assert_called_once_with(str(file))
        else:
            with patch("search_app.main.subprocess.Popen") as opener:
                assert client.post(f"/api/files/{file_id}/open", json={}).status_code == 200
                assert opener.call_args.args[0][-1] == str(file)
        assert client.post("/api/search", json={"query": "test", "mode": "invalid"}).status_code == 400
        assert client.post("/api/index", json={"folder": str(corpus)}, headers={"Origin": "https://other.example"}).status_code == 403


def test_metrics():
    actual = metrics(["wrong", "a", "b"], {"a", "b"})
    assert actual["precision_at_5"] == 0.4
    assert actual["recall_at_10"] == 1
    assert actual["mrr"] == 0.5
    assert actual["ndcg_at_10"] == pytest.approx(0.6934264)


def test_dataset_and_judgments(tmp_path):
    import json
    dataset = tmp_path / "dataset"
    generate(dataset, scale=200)
    queries = json.loads((dataset / "queries.json").read_text())
    labels = load_labels(dataset / "judgments.csv", queries, dataset / "controlled")
    assert len(labels) == 20
    assert len(list(path for path in (dataset / "controlled").rglob("*") if path.is_file())) == 55
    assert all(len(relevant) == 1 for relevant in labels.values())


@pytest.mark.skipif(not (MODEL_DIR / "modules.json").is_file(), reason="Pretrained model not installed (optional for local tests)")
def test_real_semantic_retrieval(tmp_path):
    corpus = tmp_path / "source"
    corpus.mkdir()
    (corpus / "incident.txt").write_text("A vehicle collision occurred at the intersection.")
    (corpus / "recipe.txt").write_text("Bake sourdough bread in a hot oven.")
    service = SearchService(tmp_path / "real_index")
    try:
        with TestClient(create_app(service)) as client:
            assert client.post("/api/index", json={"folder": str(corpus)}).json()["errors"] == []
            assert not client.post("/api/search", json={"query": "car accident", "mode": "keyword"}).json()["results"]
            for mode in ("semantic", "hybrid"):
                result = client.post("/api/search", json={"query": "car accident", "mode": mode})
                assert result.json()["results"][0]["name"] == "incident.txt"
            assert client.get("/api/status").json()["vectors"] == 2
    finally:
        service.close()


def test_ci_requires_real_model():
    """Fail CI rather than silently skipping real-model coverage."""
    if os.environ.get("REQUIRE_REAL_MODEL_TEST") == "1":
        assert (MODEL_DIR / "modules.json").is_file(), (
            "CI must download all-MiniLM-L6-v2 before running tests."
        )
