# User stories and acceptance criteria

| ID | User story | Acceptance criteria | Verification |
|---|---|---|---|
| US-01 | As a user, I want to index a local folder containing mixed file formats. | Supported readable files have metadata and extracted content; unsupported files are ignored; errors identify the source path. | Parser and integration tests |
| US-02 | As a user, I want to search screenshots and scanned documents. | Clear test images achieve at least 90% character accuracy and distinctive extracted terms are searchable. | OCR tests |
| US-03 | As a user, I want to find exact terms and filenames. | FTS5 returns ranked file results for known text and names. | Keyword search tests |
| US-04 | As a user, I want to find files using similar meaning. | A pretrained Sentence Transformer generates normalized embeddings; FAISS retrieves the known conceptual match; each returned ID maps to its original file. | Real-model test |
| US-05 | As a user, I want one combined search option. | RRF combines the file-level rankings correctly; duplicates are removed. | RRF and hybrid tests |
| US-06 | As a user, I want to narrow my results. | Type and folder filters exclude files outside the selected scope in every retrieval mode. | Filter tests |
| US-07 | As a user, I want my index to reflect file changes. | Unchanged content is not re-embedded; added or modified files are processed; renamed or deleted files leave no stale results; SQLite and FAISS agree. | Incremental and persistence tests |
| US-08 | As a user, I want to see and open the correct document. | Result includes file name, path, rank, score and snippet; the open action resolves the indexed file ID; missing files give an error. | API and manual UI checks |
| US-09 | As a user, I want personal data to remain on my computer. | After installing dependencies and the model, the core workflow operates locally; other origins cannot trigger index operations. | Offline model loading and API tests |
| US-10 | As a developer, I want to compare retrieval approaches fairly. | All modes use the same queries and reference judgments; metric calculations are reproducible; large-corpus tests report performance separately. | Evaluation script and metric tests |
| US-11 | As a developer, I want repeatable regression checks. | Automated tests run locally with pytest and are configured to run on GitHub for pushes and pull requests. | CI workflow |

## Backlog

| Priority | Task | Status |
|---|---|---|
| Later | Review every synthetic judgment against the source file; extend with openly licensed and non-sensitive real examples | Planned |
| Later | Repeat controlled and scale benchmarks on the target Windows machine | Planned |
| Later | Tweak chunk sizes and model selection, keeping comparison settings fixed within each run | Planned |
| Later | Improve long-running indexing feedback and cancellation | Planned |
| Optional | Experiment with visual-image retrieval using CLIP | Out of core scope |
