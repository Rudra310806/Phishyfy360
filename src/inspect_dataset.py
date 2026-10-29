
from pathlib import Path

DATASET_PATH = Path(r"C:\Phish360Data\phish360\Phish360")

def main():
    print("=" * 55)
    print("PHISHIFY360 - DATASET INSPECTION")
    print("=" * 55)

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        return

    for split in ("trainval", "test"):
        split_path = DATASET_PATH / split
        print(f"\n{split.upper()}")

        if not split_path.is_dir():
            print(f"Missing folder: {split_path}")
            continue

        samples = [
            p for p in split_path.iterdir()
            if p.is_dir()
        ]

        print(f"Sample folders: {len(samples)}")

        for sample in samples[:3]:
            print(f"\n  Sample: {sample.name}")

            for category in (
                "Label", "URL", "RAW-HTML", "SCREEN-SHOT"
            ):
                folder = sample / category

                if folder.is_dir():
                    files = list(folder.iterdir())
                    print(f"    {category}: {len(files)} file(s)")
                    for file in files[:2]:
                        print(f"      {file.name}")

    print("\nInspection complete.")

if __name__ == "__main__":
    main()
