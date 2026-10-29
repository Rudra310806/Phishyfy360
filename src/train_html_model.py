
from pathlib import Path
from collections import Counter
import random

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATASET = Path(r"C:\Phish360Data\phish360\Phish360")
MODEL_PATH = Path("models/html_model.joblib")

SEED = 42
MAX_HTML_CHARS = 100_000

random.seed(SEED)


def get_label(folder_name):
    name = folder_name.lower()

    # Official Phish360 folder naming convention.
    if name.startswith("l") and name.endswith("_legitimate"):
        return 0

    if name.startswith("p") and "_" in name:
        return 1

    return None


def load_dataset():
    root = DATASET / "trainval"

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    texts = []
    labels = []
    skipped = 0

    folders = list(root.iterdir())
    print(f"Scanning {len(folders)} dataset folders...", flush=True)

    for index, folder in enumerate(folders, start=1):
        if not folder.is_dir():
            continue

        label = get_label(folder.name)

        if label is None:
            continue

        html_path = folder / "RAW-HTML" / "index.html"

        if not html_path.is_file():
            skipped += 1
            continue

        try:
            with html_path.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as file:
                html = file.read(MAX_HTML_CHARS)

            if not html.strip():
                skipped += 1
                continue

            texts.append(html)
            labels.append(label)

        except (OSError, UnicodeError) as error:
            skipped += 1
            print(
                f"Skipped {html_path}: {error}",
                flush=True,
            )

        if index % 500 == 0:
            print(
                f"Scanned {index}/{len(folders)} folders; "
                f"usable HTML files: {len(texts)}",
                flush=True,
            )

    print(f"Skipped missing or unreadable HTML files: {skipped}")
    print("Loaded class counts:", dict(Counter(labels)))

    if len(set(labels)) != 2:
        raise RuntimeError(
            "Both legitimate and phishing HTML samples are required."
        )

    return texts, labels


def main():
    print("Loading HTML dataset...", flush=True)
    texts, labels = load_dataset()

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels,
    )

    print(f"Training samples: {len(X_train)}", flush=True)
    print(f"Validation samples: {len(X_val)}", flush=True)
    print("Building TF-IDF + Logistic Regression pipeline...", flush=True)

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=50_000,
                min_df=2,
                sublinear_tf=True,
                dtype=float,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    ])

    print("Training HTML detector...", flush=True)
    model.fit(X_train, y_train)

    print("Evaluating on held-out validation data...", flush=True)
    predictions = model.predict(X_val)

    print(
        f"\nValidation accuracy: "
        f"{accuracy_score(y_val, predictions):.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Phishing"],
            zero_division=0,
        )
    )

    print("Confusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(y_val, predictions, labels=[0, 1]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nHTML model saved to: {MODEL_PATH.resolve()}", flush=True)
    print("HTML detector training complete!", flush=True)


if __name__ == "__main__":
    main()
