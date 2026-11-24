"""SQLite metadata, text, FTS5 and durable chunk storage."""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    extension TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_ns INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    UNIQUE(file_id, position)
);
CREATE TABLE IF NOT EXISTS folders (path TEXT PRIMARY KEY);
CREATE VIRTUAL TABLE IF NOT EXISTS file_fts USING fts5(
    name, path, content, tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS remove_fts AFTER DELETE ON files BEGIN
    DELETE FROM file_fts WHERE rowid = old.id;
END;
"""


class Database:
    """Keep all durable records in one transactional SQLite database."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)

    def save_file(self, path: Path, digest: str, text: str, chunks: list[str], vectors) -> None:
        """Replace a file and both its text and vector records in one transaction."""
        stat = path.stat()
        with self.connection:
            self.connection.execute("DELETE FROM files WHERE path = ?", (str(path),))
            cursor = self.connection.execute(
                """INSERT INTO files(path, name, extension, size_bytes, modified_ns, sha256, content)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (str(path), path.name, path.suffix.lower(), stat.st_size, stat.st_mtime_ns, digest, text),
            )
            file_id = cursor.lastrowid
            self.connection.execute(
                "INSERT INTO file_fts(rowid, name, path, content) VALUES (?, ?, ?, ?)",
                (file_id, path.name, str(path), text),
            )
            self.connection.executemany(
                "INSERT INTO chunks(file_id, position, text, embedding) VALUES (?, ?, ?, ?)",
                [(file_id, position, chunk, vector.tobytes())
                 for position, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))],
            )

    def files(self) -> list[sqlite3.Row]:
        """Return all indexed file metadata and content."""
        return self.connection.execute("SELECT * FROM files ORDER BY id").fetchall()

    def chunks(self) -> list[sqlite3.Row]:
        """Return the durable chunk-to-file mapping."""
        return self.connection.execute("SELECT * FROM chunks ORDER BY id").fetchall()

    def delete_file(self, path: str) -> None:
        """Remove the file, cascading to its chunks and FTS entry."""
        with self.connection:
            self.connection.execute("DELETE FROM files WHERE path = ?", (path,))

    def close(self) -> None:
        self.connection.close()
