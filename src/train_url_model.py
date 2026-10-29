
from pathlib import Path
from collections import Counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATASET = Path(r"C:\Phish360Data\phish360\Phish360")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "url_model.joblib"

def read_text(path):
    return path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).strip()

def get_label(folder_name):
    suffix = folder_name.rsplit("_", 1)[-1].lower()

    if suffix == "legitimate":
        return 0
    if suffix == "phish" or suffix not in ("", "unknown"):
        # Phish360 training folders use target-brand suffixes
        # for phishing samples. Verify counts before accepting.
        return 1

    return None

def load_samples(split):
    root = DATASET / split
    urls, labels = [], []
    skipped = 0
    counts = Counter()

    for sample in root.iterdir():
        if not sample.is_dir():
            continue

        label = get_label(sample.name)
        url_path = sample / "URL" / "url.txt"

        if label is None or not url_path.is_file():
            skipped += 1
            continue

        url = read_text(url_path)
        if not url:
            skipped += 1
            continue

        urls.append(url)
        labels.append(label)
        counts[label] += 1

    print(f"\n{split}: {len(urls)} usable samples")
    print(f"Class counts (0=legitimate, 1=phishing): {dict(counts)}")
    print(f"Skipped: {skipped}")

    return urls, labels

def main():
    urls, labels = load_samples("trainval")

    if len(set(labels)) != 2:
        raise ValueError("Expected both legitimate and phishing samples.")

    X_train, X_val, y_train, y_val = train_test_split(
        urls,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=150_000,
                sublinear_tf=True,
                dtype=float,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining URL classifier...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    print("\n--- VALIDATION RESULTS ---")
    print(f"Accuracy: {accuracy_score(y_val, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_val,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, predictions, labels=[0, 1]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    main()
