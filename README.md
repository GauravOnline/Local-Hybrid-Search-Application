# Local Hybrid Search

A local information retrieval system for personal files. The application combines traditional full-text search with semantic vector search and OCR so that documents can be found even when their exact wording or filename is unknown.

**Author:** Gaurav Shienhmar

## Features

- Recursively index local folders containing PDF, DOCX, TXT, PNG, JPG and JPEG files.
- Extract printed text from screenshots and scanned PDF pages using Tesseract.
- Keep file metadata, content and chunk embeddings in SQLite.
- Run keyword (FTS5/BM25), semantic (Sentence Transformers/FAISS) or hybrid (Reciprocal Rank Fusion) retrieval.
- Filter by file type and indexed folder; view the rank, score, snippet and original path.
- Refresh folders to detect additions, edits, renames and deletions using SHA-256 hashes.
- Compare retrieval methods with a controlled dataset and measure indexing, storage and query performance.

## 1. Setup on Windows

Install **64-bit Python 3.11** and Tesseract OCR with English language data. Add the Tesseract installation folder to PATH and open a new PowerShell window.

Check the installations:

```powershell
py -3.11 --version
tesseract --version
tesseract --list-langs
```

If Tesseract is installed but not on PATH, set its executable path for the current window:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Open PowerShell in the extracted `LocalHybridSearch` folder (the folder containing this README).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m scripts.download_model
```

The first installation and model download require internet access. The model is saved under `models/all-MiniLM-L6-v2`. After setup, indexing and searching use local files and the local model; neither uploads personal data nor calls an external API.

## 2. Run the application

```powershell
.\.venv\Scripts\python.exe -m uvicorn search_app.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in a browser. Keep the PowerShell window running and press Ctrl+C to stop it.

1. Paste a full local folder path into **Source folders**.
2. Select **Index / refresh folder**. The first scan may take longer as OCR and embeddings are generated.
3. Enter a query, choose a retrieval mode, optionally apply filters, and search.
4. Use **Open file** to open the original document through the operating system.
5. Refresh the source folder after adding, editing, renaming or deleting files. Unchanged files are hashed but not extracted or embedded again.

## 3. Try a controlled dataset

The distribution archive includes a sample dataset under `datasets/`. For a fresh source checkout without generated datasets, run:

```powershell
.\.venv\Scripts\python.exe -m scripts.generate_dataset
```

The dataset contains 55 controlled synthetic files, 20 queries and 1,100 draft binary relevance judgments, plus a separate 250-file collection for performance testing. Use the absolute path to `datasets/controlled` in the application. To generate a fresh copy without overwriting the included data, pass `--output datasets_new`.

| Query | What to observe |
|---|---|
| Orion-482 | Exact reference found through keyword retrieval |
| car accident | Semantic retrieval finds a document about a vehicle collision |
| budget_q7 | Metadata search finds a file by its filename |
| LUMEN-204 | OCR text makes an image searchable |
| car accident (hybrid) | RRF combines evidence from the two retrieval methods |

To check incremental indexing, refresh a previously indexed folder, modify a test TXT file and refresh again. The indexing report distinguishes unchanged, added, updated and removed paths.

## 4. Test and evaluate

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.evaluate --dataset datasets
.\.venv\Scripts\python.exe -m scripts.evaluate --dataset datasets --scale
```

Evaluation uses isolated temporary indexes, leaving the application's index untouched. Results are written to `evaluation_results/controlled.json` and `evaluation_results/scale.json`.

The synthetic judgments describe known generated content and must be reviewed against the original files before being used as final experimental ground truth. The larger corpus measures system performance and known-file discoverability only; it is not exhaustively labelled.

## 5. Project layout

| Path | Responsibility |
|---|---|
| `search_app/extraction.py` | Hashing, parsing, OCR and chunking |
| `search_app/database.py` | Metadata, FTS5 and durable embeddings |
| `search_app/vectors.py` | Sentence Transformer and FAISS |
| `search_app/service.py` | Incremental indexing, ranking and filters |
| `search_app/main.py` | Local FastAPI endpoints |
| `search_app/static/` | HTML, CSS and JavaScript interface |
| `scripts/` | Model setup, dataset generation and evaluation |
| `tests/` | Unit, functional and integration testing |
| `docs/` | Design decisions, requirements and demonstration |
| `.github/workflows/tests.yml` | Automated regression tests |

## 6. Testing and continuous integration

The project uses pytest for unit and integration testing. GitHub Actions runs the tests on pushes and pull requests using `.github/workflows/tests.yml`. The workflow installs Tesseract and Python dependencies, downloads the pretrained embedding model, then executes the test suite, including the semantic integration test.

The generated datasets, pretrained model, local indexes, and evaluation outputs are excluded from version control. Datasets can be regenerated using `python -m scripts.generate_dataset`. CI execution should be verified using the workflow run results in the repository.

## Notes

- Use one running server process for a given data directory.
- The application stores extracted personal text and embeddings locally in `data/`. This is not an encrypted vault; protect that folder as you would the source files.
- OCR is intended for readable printed English. Handwriting and low-resolution images can produce incomplete text.
- The semantic model searches extracted text, not visual objects. Scans without a text layer are OCR processed; mixed PDF pages with an existing text layer use that layer.
- The prototype uses a flat, exact FAISS index suitable for a few hundred files. It keeps the durable embeddings in SQLite and rebuilds the derived vector index when records change. This avoids regenerating embeddings for unchanged files.
- Changing the embedding model or chunking configuration requires a fresh index; use a different `SEARCH_DATA_DIR` to preserve the original.
- The application does not alter the original files. Refreshing a folder only changes its local index.

On Linux/macOS, use `python3 -m venv .venv`, `source .venv/bin/activate`, and the same module commands with `python`. Install Tesseract separately.
