
from pathlib import Path
from collections import Counter

DATASET = Path(r"C:\Phish360Data\phish360\Phish360")

for split in ("trainval", "test"):
    root = DATASET / split
    counts = Counter()
    examples = {}

    for sample in root.iterdir():
        if not sample.is_dir():
            continue

        label_file = sample / "Label" / "label.txt"
        if not label_file.is_file():
            continue

        raw = label_file.read_text(
            encoding="utf-8-sig", errors="replace"
        ).strip()

        counts[raw] += 1
        examples.setdefault(raw, sample.name)

    print(f"\n--- {split.upper()} ---")
    print(f"Distinct raw labels: {len(counts)}")

    for label, count in counts.most_common():
        print(f"{label!r}: {count} | example: {examples[label]}")
