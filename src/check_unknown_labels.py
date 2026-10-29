
from pathlib import Path

DATASET = Path(r"C:\Phish360Data\phish360\Phish360")

for split in ("trainval", "test"):
    root = DATASET / split
    print(f"\n--- {split.upper()} ---")

    shown = 0

    for sample in root.iterdir():
        if not sample.is_dir():
            continue

        label_file = sample / "Label" / "label.txt"
        if not label_file.is_file():
            continue

        label = label_file.read_text(
            encoding="utf-8-sig", errors="replace"
        ).strip()

        if label == "?":
            print(f"Sample: {sample.name}")
            print(f"Label: {label!r}")

            url_file = sample / "URL" / "url.txt"
            if url_file.is_file():
                print("URL:", url_file.read_text(
                    encoding="utf-8-sig", errors="replace"
                ).strip())

            shown += 1
            if shown >= 5:
                break
