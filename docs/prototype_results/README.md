# Prototype verification

## Test environment

- Python 3.12.14, Linux x86-64, AMD EPYC 9V74 CPU (9 logical processors available).
- Sentence Transformers 3.4.1, all-MiniLM-L6-v2 downloaded locally, CPU inference (up to 4 threads).
- FAISS 1.16.0, SQLite 3.53.1 and Tesseract OCR.
- Model loading and retrieval ran with Hugging Face offline settings enabled.

## Automated checks

15 pytest tests passed, including the real pretrained model test.

The tests validate text extraction from TXT, PDF and DOCX, OCR from PNG, JPG and scanned PDF (at least 90% character accuracy for clear test samples), SHA-256 hashing, chunking, RRF, filtering, deduplication, incremental indexing, persistence, API behavior and ranking metrics. The file-open test checks that the correct path is passed to the operating system's opener; a manual check on the target Windows desktop remains necessary.

## Initial controlled retrieval experiment

The corpus contains 55 synthetic files, with 20 queries and one intended relevant file per query. The draft judgments were determined by the authored content; they are not independently reviewed final reference labels.

| Method | Precision@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Keyword | 0.15 | 0.75 | 0.75 | 0.75 |
| Semantic | 0.20 | 1.00 | 1.00 | 1.00 |
| Hybrid | 0.20 | 1.00 | 0.93 | 0.95 |

Keyword retrieval found the intended result for all exact, metadata and OCR queries but none of the conceptual searches. Semantic retrieval found the expected conceptual files, demonstrating that the query and document do not need to share the same words. Hybrid retrieved the known files but ranked some conceptual matches lower because irrelevant lexical matches also contributed to RRF. This is an early observation from a deliberately simple dataset, not evidence that one mode will always be better.

Precision@5 cannot exceed 0.20 in this corpus because each query has one intended relevant file.

## Initial scale experiment

| Measurement | 55 files | 250 files |
|---|---:|---:|
| Initial indexing (seconds) | 4.951 | 7.227 |
| No-change refresh (seconds) | 0.006 | 0.025 |
| Unchanged files skipped | 55 | 250 |
| Index storage (bytes) | 298,002 | 1,123,370 |
| Indexing errors | 0 | 0 |

| Method | 250-file warm median (ms) | 250-file p95 (ms) | Known file found @10 |
|---|---:|---:|---:|
| Keyword | 0.32 | 0.50 | 75% |
| Semantic | 7.04 | 8.51 | 100% |
| Hybrid | 7.93 | 11.98 | 100% |

All 180 measured warm queries on the scale collection completed within five seconds. Known-file discovery is not exhaustive recall. These measurements exclude the model download, browser rendering and server overhead; the performance experiment measures the retrieval service itself. Index storage excludes the pretrained model and original files.

The JSON files in this directory contain the raw observed rankings, query latencies, indexing reports and category-level results. Rerun the experiment on the target computer before treating these timings as final results.

## Remaining manual checks

- Run the local browser interface on Windows and open each supported file type in its native application.
- Verify model loading and search with the network physically disconnected.
- Review every draft relevance judgment, add more diverse source files and repeat the experiments.
