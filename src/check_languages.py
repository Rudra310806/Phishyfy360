
from pathlib import Path
from collections import Counter

from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory, LangDetectException

DATASET = Path(r"C:\Phish360Data\phish360\Phish360")
ROOT = DATASET / "trainval"

MAX_SAMPLES = 500
MAX_HTML_CHARS = 100_000

DetectorFactory.seed = 42


def get_label(folder_name):
    name = folder_name.lower()

    if name.startswith("l") and name.endswith("_legitimate"):
        return "legitimate"

    if name.startswith("p") and "_" in name:
        return "phishing"

    return None


def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")

    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()

    return " ".join(soup.stripped_strings)


def main():
    if not ROOT.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {ROOT}")

    folders = [
        folder for folder in ROOT.iterdir()
        if folder.is_dir() and get_label(folder.name)
    ]

    print(f"Found {len(folders)} labelled folders.")
    print(f"Checking up to {MAX_SAMPLES} HTML files...\n")

    import random

    rng = random.Random(42)

    legitimate_folders = [
        folder for folder in folders
        if get_label(folder.name) == "legitimate"
    ]

    phishing_folders = [
        folder for folder in folders
        if get_label(folder.name) == "phishing" 
    ]

# Sample up to 250 folders from each class.
    per_class = MAX_SAMPLES // 2

    selected_folders = (
        rng.sample(legitimate_folders, min(per_class, len(legitimate_folders)))
        + rng.sample(phishing_folders, min(per_class, len(phishing_folders)))       
    )

    random_order = selected_folders
    rng.shuffle(random_order)
    language_counts = Counter()
    class_languages = {
        "legitimate": Counter(),
        "phishing": Counter(),
    }

    processed = 0
    skipped = 0

    for folder in random_order:
        if processed >= MAX_SAMPLES:
            break

        html_path = folder / "RAW-HTML" / "index.html"

        try:
            with html_path.open(
                "r", encoding="utf-8", errors="replace"
            ) as file:
                html = file.read(MAX_HTML_CHARS)

            text = extract_text(html)

            # Avoid assigning languages to tiny or nearly empty pages.
            if len(text) < 100:
                skipped += 1
                continue

            language = detect(text)

            language_counts[language] += 1
            class_languages[get_label(folder.name)][language] += 1
            processed += 1

            if processed % 100 == 0:
                print(f"Processed {processed}/{MAX_SAMPLES} usable pages...")

        except (OSError, LangDetectException, ValueError):
            skipped += 1

    print("\n========== LANGUAGE AUDIT ==========")
    print("Successfully analyzed:", processed)
    print("Skipped files/pages:", skipped)

    print("\nOverall language counts:")
    for language, count in language_counts.most_common():
        print(f"{language}: {count}")

    print("\nLegitimate page languages:")
    for language, count in class_languages["legitimate"].most_common():
        print(f"{language}: {count}")

    print("\nPhishing page languages:")
    for language, count in class_languages["phishing"].most_common():
        print(f"{language}: {count}")


if __name__ == "__main__":
    main()
