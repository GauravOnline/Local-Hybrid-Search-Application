"""Coordinate indexing, retrieval and file-level ranking."""

import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from .config import DATA_DIR, MODEL_DIR, RRF_K, SUPPORTED_TYPES
from .database import Database
from .extraction import chunk_text, extract_text, file_hash
from .vectors import SentenceEncoder, VectorIndex


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Combine rankings without comparing incompatible BM25 and cosine scores."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, file_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[file_id] = scores.get(file_id, 0.0) + 1 / (k + rank)
    return dict(sorted(scores.items(), key=lambda item: (-item[1], item[0])))


def fts_query(query: str) -> str:
    """Convert natural input into a safe FTS5 expression."""
    tokens = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
    return " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))


class SearchService:
    """Own one database and one synchronized vector index."""

    def __init__(self, data_dir: Path = DATA_DIR, encoder=None):
        self.data_dir = data_dir.resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database = Database(self.data_dir / "search.sqlite3")
        self.encoder = encoder or SentenceEncoder()
        self.vector_index = VectorIndex(self.encoder.dimension, self.data_dir / "vectors.faiss")
        self.lock = threading.RLock()
        self._dirty = True
        self._sync_vectors()

    def _sync_vectors(self) -> None:
        if self._dirty:
            self.vector_index.rebuild(self.database.chunks())
            self._dirty = False

    def _discover(self, root: Path) -> set[Path]:
        """Complete discovery before deleting records, so an inaccessible folder is safe."""
        if not root.is_dir():
            raise ValueError(f"Folder does not exist: {root}")
        discovered = set()

        def on_error(error: OSError) -> None:
            raise error

        excluded = [self.data_dir, MODEL_DIR.resolve()]
        for directory, folders, files in os.walk(root, onerror=on_error, followlinks=False):
            current = Path(directory)
            folders[:] = sorted(
                name for name in folders
                if not (current / name).is_symlink()
                and not any((current / name).resolve().is_relative_to(path) for path in excluded)
            )
            for name in files:
                path = current / name
                if not path.is_symlink() and path.suffix.lower() in SUPPORTED_TYPES:
                    discovered.add(path.resolve())
        return discovered

    def index_folder(self, folder: str | Path) -> dict:
        """Hash files, process only changed content and remove missing paths."""
        start = time.perf_counter()
        root = Path(folder).expanduser().resolve()
        if any(root.is_relative_to(path) for path in (self.data_dir, MODEL_DIR.resolve())):
            raise ValueError("Select a source folder outside the application data/model directories.")
        with self.lock:
            paths = self._discover(root)
            previous = {row["path"]: row for row in self.database.files()
                        if Path(row["path"]).is_relative_to(root)}
            report = {"folder": str(root), "discovered": len(paths), "added": 0,
                      "updated": 0, "unchanged": 0, "removed": 0, "errors": [], "warnings": []}
            with self.database.connection:
                self.database.connection.execute("INSERT OR IGNORE INTO folders VALUES (?)", (str(root),))
            try:
                for path in sorted(paths):
                    old = previous.get(str(path))
                    try:
                        digest = file_hash(path)
                        if old and old["sha256"] == digest:
                            stat = path.stat()
                            with self.database.connection:
                                self.database.connection.execute(
                                    "UPDATE files SET modified_ns=?, size_bytes=? WHERE id=?",
                                    (stat.st_mtime_ns, stat.st_size, old["id"]),
                                )
                            report["unchanged"] += 1
                            continue
                        text = extract_text(path)
                        if not text:
                            report["warnings"].append({"path": str(path), "message": "No extracted text; metadata only."})
                        chunks = chunk_text(text) or [""]
                        # Filename and folder provide metadata context for semantic queries.
                        inputs = [f"{path.stem} {path.parent.name} {chunk}" for chunk in chunks]
                        vectors = np.asarray(self.encoder.encode(inputs), dtype=np.float32)
                        if vectors.shape != (len(chunks), self.encoder.dimension) or not np.isfinite(vectors).all():
                            raise ValueError("Unexpected embedding shape or values.")
                        if file_hash(path) != digest:
                            raise RuntimeError("File changed during indexing. Retry the scan.")
                        self.database.save_file(path, digest, text, chunks, vectors)
                        self._dirty = True
                        report["updated" if old else "added"] += 1
                    except Exception as error:
                        report["errors"].append({"path": str(path), "message": str(error)})
                        if old:
                            # An unprocessed replacement must not return stale content.
                            self.database.delete_file(str(path))
                            self._dirty = True
                for path in sorted(previous.keys() - {str(path) for path in paths}):
                    self.database.delete_file(path)
                    self._dirty = True
                    report["removed"] += 1
            finally:
                self._sync_vectors()
            report["seconds"] = round(time.perf_counter() - start, 3)
            return report

    def search(self, query: str, mode: str = "hybrid", extension: str = "",
               folder: str = "", limit: int = 10) -> dict:
        """Retrieve and rank unique files, applying the same filters in each mode."""
        start = time.perf_counter()
        if mode not in {"keyword", "semantic", "hybrid"}:
            raise ValueError("Unknown retrieval mode.")
        if not query.strip() or not 1 <= limit <= 100:
            raise ValueError("Enter a query and a limit between 1 and 100.")
        with self.lock:
            self._sync_vectors()
            selected_folder = Path(folder).expanduser().resolve() if folder else None
            files = {row["id"]: row for row in self.database.files()
                     if (not extension or row["extension"] == extension.lower())
                     and (not selected_folder or Path(row["path"]).is_relative_to(selected_folder))}
            keyword: dict[int, tuple[float, str]] = {}
            semantic: dict[int, tuple[float, str]] = {}
            expression = fts_query(query)
            if mode != "semantic" and expression and files:
                rows = self.database.connection.execute(
                    """SELECT rowid, bm25(file_fts, 3.0, 1.0, 1.0) AS score,
                       snippet(file_fts, 2, '', '', ' … ', 32) AS snippet
                       FROM file_fts WHERE file_fts MATCH ? ORDER BY score, rowid""", (expression,),
                )
                keyword = {row["rowid"]: (-row["score"], row["snippet"])
                           for row in rows if row["rowid"] in files}
            if mode != "keyword" and files:
                embedding = self.encoder.encode([query])[0]
                chunks = {row["id"]: row for row in self.database.chunks()}
                for chunk_id, score in self.vector_index.search(embedding):
                    chunk = chunks[chunk_id]
                    file_id = chunk["file_id"]
                    if file_id in files and file_id not in semantic:
                        # The highest-scoring chunk determines the file's semantic rank.
                        semantic[file_id] = (score, chunk["text"])
            if mode == "hybrid":
                scores = reciprocal_rank_fusion([list(keyword), list(semantic)])
            else:
                scores = {key: value[0] for key, value in
                          (keyword if mode == "keyword" else semantic).items()}
            results = []
            keyword_ranks = {file_id: rank for rank, file_id in enumerate(keyword, 1)}
            semantic_ranks = {file_id: rank for rank, file_id in enumerate(semantic, 1)}
            for rank, (file_id, score) in enumerate(list(scores.items())[:limit], start=1):
                row = files[file_id]
                snippet = (keyword.get(file_id) or semantic[file_id])[1]
                results.append({
                    "id": file_id, "name": row["name"], "path": row["path"],
                    "extension": row["extension"], "size_bytes": row["size_bytes"],
                    "modified_ns": row["modified_ns"], "rank": rank, "score": round(score, 7),
                    "snippet": snippet[:600] or "(Metadata match; no extracted text.)",
                    "keyword_rank": keyword_ranks.get(file_id),
                    "semantic_rank": semantic_ranks.get(file_id),
                })
            return {"mode": mode, "results": results, "latency_ms": round((time.perf_counter() - start) * 1000, 3)}

    def status(self) -> dict:
        """Expose index counts and storage usage without reading source files."""
        with self.lock:
            self._sync_vectors()
            connection = self.database.connection
            return {
                "files": connection.execute("SELECT count(*) FROM files").fetchone()[0],
                "chunks": connection.execute("SELECT count(*) FROM chunks").fetchone()[0],
                "vectors": self.vector_index.index.ntotal,
                "folders": [row[0] for row in connection.execute("SELECT path FROM folders ORDER BY path")],
                "storage_bytes": sum(path.stat().st_size for path in self.data_dir.iterdir() if path.is_file()),
            }

    def source(self, file_id: int) -> Path:
        """Resolve only previously indexed files for opening."""
        with self.lock:
            row = self.database.connection.execute("SELECT path FROM files WHERE id=?", (file_id,)).fetchone()
            if not row or not Path(row["path"]).is_file():
                raise FileNotFoundError("The source file no longer exists. Re-index the folder.")
            return Path(row["path"])

    def close(self) -> None:
        self.database.close()
