# System design

## Data flow

1. A folder is scanned for supported file types. The SHA-256 hash identifies unchanged content.
2. The file parser extracts text directly or through OCR. The content is split into overlapping word chunks.
3. Each chunk is encoded using all-MiniLM-L6-v2 (384 dimensions, CPU, normalized vectors).
4. SQLite stores metadata, full text and chunk embeddings. FTS5 indexes the file name, path and content. FAISS stores vectors keyed by SQLite chunk ID.
5. The retrieval service returns file-level keyword or semantic rankings, or combines both with RRF.
6. The FastAPI interface displays results and opens source files on the local machine.

## Data model

| Entity | Key | Main attributes |
|---|---|---|
| files | id | unique path, name, extension, size, modified time, SHA-256, content |
| chunks | id | file_id, position, text, float32 embedding |
| file_fts | rowid = files.id | name, path, content |
| folders | path | indexed folder path |
| FAISS | chunks.id | normalized vector |

Each file has one or more chunks. SQLite is the authoritative store: deleting a file cascades to its chunks and an FTS trigger removes the lexical record. A changed file is replaced in a single transaction. FAISS is rebuilt from the committed embeddings when records change and at startup; the resulting index file is written through a temporary file before replacement. No model inference is needed to rebuild unchanged embeddings.

## Retrieval decisions

- Keyword search uses safely quoted tokens with OR matching and a higher weight on filenames. FTS5 BM25 is ascending, so the sign is reversed for display.
- Semantic search uses an exact FAISS inner-product index over normalized vectors. A file's highest-scoring chunk determines its rank. Multiple chunks from the same file contribute only one file-level result.
- Chunks contain up to 160 words with 30 words of overlap. Names and parent folders provide semantic metadata context. The pretrained model has a token limit; unusually long words or metadata can cause truncation.
- RRF uses k=60. Each candidate receives 1/(60 + rank) for each ranking containing it. This combines ranks without treating BM25 and cosine scores as comparable.
- File-type and folder filters are applied before file-level fusion. All eligible vector candidates are retrieved in this prototype to avoid losing valid filtered results.
- A numeric score is specific to its retrieval mode; it is not a percentage or calibrated probability.

## Consistency and privacy

A single service lock serializes index updates and searches. Discovery completes before missing records are deleted. Failed replacements are removed from the index and reported, preventing outdated search results. A later refresh can retry them. Symlinks and application data/model folders are skipped.

The server listens on loopback, serves its own static assets, and rejects cross-origin requests. The model is downloaded only during setup and explicitly loaded from local storage thereafter. This design targets a single trusted local user, not a public or multi-user server.

## Evaluation

The controlled corpus includes 20 queries (5 exact, 5 conceptual, 5 metadata, 5 OCR), 55 files and complete draft judgments. Precision@5 uses a denominator of five, Recall@10 uses the number of relevant files, MRR uses the full controlled ranking, and nDCG@10 uses binary relevance. The scale corpus reports warm latency, storage, indexing and known-file discovery at ten results without claiming exhaustive recall.

The implementation intentionally leaves model fine-tuning, visual CLIP retrieval, automatic background indexing, approximate FAISS indexes, and richer relevance judgments as later extensions.
