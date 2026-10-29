
from pathlib import Path
from collections import Counter

DATASET = Path(r"C:\Phish360Data\phish360\Phish360")

for split in ("trainval", "test"):
    root = DATASET / split
    samples = [p for p in root.iterdir() if p.is_dir()]

    print(f"\n{'=' * 45}")
    print(f"{split.upper()} — {len(samples)} samples")

    suffixes = Counter(
        p.name.split("_", 1)[-1].lower()
        for p in samples
    )

    print("\nFolder-name patterns:")
    for suffix, count in suffixes.most_common(20):
        print(f"  {suffix}: {count}")

    print("\nExample folder names:")
    for sample in samples[:8]:
        print(" ", sample.name)

    print("\nExample label contents:")
    for sample in samples[:5]:
        label_file = sample / "Label" / "label.txt"
        if label_file.exists():
            value = label_file.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
            print(f"  {sample.name} -> {value!r}")
