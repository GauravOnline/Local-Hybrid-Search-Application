"""Download the compact embedding model once before using the app offline."""

from sentence_transformers import SentenceTransformer

from search_app.config import MODEL_DIR, MODEL_NAME


def main() -> None:
    if (MODEL_DIR / "modules.json").is_file():
        print(f"Model already installed: {MODEL_DIR}")
        return
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR))
    print(f"Saved model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
