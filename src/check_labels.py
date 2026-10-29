
from pathlib import Path
from collections import Counter

DATASET = Path(r"C:\Phish360Data\phish360\Phish360")

def main():
    for split in ("trainval", "test"):
        root = DATASET / split
        counts = Counter()
        label_values = Counter()
        missing_urls = 0
        missing_labels = 0

        if not root.exists():
            print(f"Missing split: {root}")
            continue

        samples = [p for p in root.iterdir() if p.is_dir()]

        for sample in samples:
            # Folder names provide a preliminary label check.
            name = sample.name.lower()
            if name.endswith("_legitimate"):
                counts["legitimate"] += 1
            elif name.endswith("_phish"):
                counts["phishing"] += 1
            else:
                counts["unknown"] += 1

            label_file = sample / "Label" / "label.txt"
            url_file = sample / "URL" / "url.txt"

            if label_file.is_file():
                value = label_file.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
                label_values[value] += 1
            else:
                missing_labels += 1

            if not url_file.is_file():
                missing_urls += 1

        print(f"\n--- {split.upper()} ---")
        print(f"Sample folders: {len(samples)}")
        print(f"Folder-name labels: {dict(counts)}")
        print(f"Actual label.txt values: {dict(label_values)}")
        print(f"Missing label files: {missing_labels}")
        print(f"Missing URL files: {missing_urls}")

if __name__ == "__main__":
    main()
