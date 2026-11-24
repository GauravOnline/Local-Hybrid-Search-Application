# Prototype demonstration

## Setup

Run the model setup, dataset generator and local FastAPI server as described in the README. Open the search interface and index the controlled dataset folder.

## Demonstration sequence

1. **Ingestion:** show the file, chunk, vector and storage counters. Index 55 controlled files and inspect the scan report.
2. **Keyword:** search `Orion-482` using Keyword mode; verify the original text and file path.
3. **Semantic:** search `car accident` using Keyword mode, then Semantic mode. Inspect the result describing a vehicle collision. The semantic result is based on similarity, rather than exact words.
4. **Hybrid:** repeat the conceptual query in Hybrid mode and inspect the component ranks. RRF uses ranks, not raw cross-model scores.
5. **OCR:** search `LUMEN-204`, filter to JPG, and open the matching source image. The text is printed in the image.
6. **Metadata:** search `budget_q7` and use the folder/type selectors to narrow results.
7. **Incremental:** refresh the same folder to observe zero new embeddings; create a TXT file with a distinctive phrase, refresh, search, rename it, refresh, then remove it and refresh again. Use a separate disposable folder for this step so the evaluation corpus stays unchanged.
8. **Persistence:** stop and restart the server; search again without re-indexing.
9. **Evaluation:** run pytest and the controlled/scale scripts. Inspect the measured results without assuming which retrieval mode will win.

## Explain the main engineering decisions

- FTS5 is the lexical baseline; it is fast and supports exact terms.
- The semantic model encodes text into normalized vectors; FAISS finds similar chunks.
- Ranking is performed at the file level so a long document does not occupy multiple results.
- OCR makes images searchable through the same pipeline.
- SQLite owns durable data; rebuilding FAISS from stored chunks keeps it consistent without re-encoding unchanged documents.
- RRF combines ranks using a simple formula that avoids comparing different scoring systems.
