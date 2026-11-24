"""Compare retrieval modes on the same corpus with fixed relevance labels."""

import argparse
import csv
import json
import math
import os
import platform
import statistics
import tempfile
from pathlib import Path

import numpy as np

from search_app.service import SearchService

MODES = ("keyword", "semantic", "hybrid")


def metrics(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    """Compute file-level binary relevance metrics, with a fixed precision denominator."""
    hits = [int(path in relevant) for path in ranked[:10]]
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(10, len(relevant))))
    return {
        "precision_at_5": sum(hits[:5]) / 5,
        "recall_at_10": sum(hits) / len(relevant) if relevant else 0.0,
        "mrr": next((1 / (i + 1) for i, path in enumerate(ranked) if path in relevant), 0.0),
        "ndcg_at_10": dcg / ideal if ideal else 0.0,
    }


def load_labels(path: Path, queries: list[dict], corpus: Path) -> dict[str, set[str]]:
    """Reject missing or duplicate judgments before computing metrics."""
    files = {file.relative_to(corpus).as_posix() for file in corpus.rglob("*") if file.is_file()}
    labels = {query["id"]: set() for query in queries}
    expected = {(query["id"], file) for query in queries for file in files}
    seen = set()
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            pair = (row["query_id"], row["file_id"])
            if pair not in expected or pair in seen or row["relevant"] not in {"0", "1"} or not row["evidence"].strip():
                raise ValueError(f"Invalid judgment: {pair}")
            seen.add(pair)
            if row["relevant"] == "1":
                labels[pair[0]].add(pair[1])
    if seen != expected:
        raise ValueError("The controlled judgments do not cover every query-file pair.")
    return labels


def run_experiment(dataset: Path, scale: bool, repetitions: int) -> dict:
    """Measure indexing, no-change refresh, index storage and warm query timings."""
    root = (dataset / ("scale" if scale else "controlled")).resolve()
    queries = json.loads((dataset / "queries.json").read_text(encoding="utf-8"))
    labels = None if scale else load_labels(dataset / "judgments.csv", queries, root)
    with tempfile.TemporaryDirectory(prefix="hybrid_eval_") as directory:
        service = SearchService(Path(directory))
        try:
            first = service.index_folder(root)
            second = service.index_folder(root)
            status = service.status()
            if first["errors"] or second["errors"] or status["files"] != first["discovered"]:
                raise RuntimeError(f"Incomplete indexing: {first['errors'] + second['errors']}")
            rows = []
            for mode in MODES:
                service.search(queries[0]["query"], mode)
                for query in queries:
                    times = []
                    for _ in range(repetitions):
                        result = service.search(query["query"], mode, limit=10 if scale else 100)
                        times.append(result["latency_ms"])
                    ranked = [Path(item["path"]).relative_to(root).as_posix() for item in result["results"]]
                    row = {"query_id": query["id"], "category": query["category"], "mode": mode,
                           "ranked_paths": ranked, "latencies_ms": times,
                           "known_file_found": query["known_file"] in ranked[:10]}
                    if labels is not None:
                        row.update(metrics(ranked, labels[query["id"]]))
                    rows.append(row)
            summary = {}
            categories = ["all"] + sorted({query["category"] for query in queries})
            for mode in MODES:
                summary[mode] = {}
                for category in categories:
                    group = [row for row in rows if row["mode"] == mode
                             and (category == "all" or row["category"] == category)]
                    times = [value for row in group for value in row["latencies_ms"]]
                    stats = {"median_ms": statistics.median(times), "p95_ms": float(np.percentile(times, 95)),
                             "within_5s_fraction": sum(value <= 5000 for value in times) / len(times),
                             "known_file_rate": statistics.mean(row["known_file_found"] for row in group)}
                    if labels is not None:
                        for key in ("precision_at_5", "recall_at_10", "mrr", "ndcg_at_10"):
                            stats[key] = statistics.mean(row[key] for row in group)
                    summary[mode][category] = stats
            # Report dataset-relative locations rather than machine-specific source paths.
            first["folder"] = second["folder"] = root.name
            status["folders"] = [root.name]
            processor = platform.processor()
            if not processor and Path("/proc/cpuinfo").is_file():
                processor = next((line.split(":", 1)[1].strip()
                                  for line in Path("/proc/cpuinfo").read_text().splitlines()
                                  if line.startswith("model name")), "Unknown")
            return {"dataset": "scale" if scale else "controlled", "hardware": platform.platform(),
                    "processor": processor, "logical_cpus": os.cpu_count(),
                    "python": platform.python_version(), "repetitions": repetitions,
                    "model": "all-MiniLM-L6-v2 (CPU)", "inference_threads": min(4, os.cpu_count() or 1),
                    "label_status": "synthetic draft judgments" if not scale else "known-file checks only",
                    "initial_index": first, "refresh": second, "status": status,
                    "summary": summary, "queries": rows}
        finally:
            service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets"))
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results"))
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("Repetitions must be positive.")
    result = run_experiment(args.dataset, args.scale, args.repetitions)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / ("scale.json" if args.scale else "controlled.json")
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved results to {destination}")
