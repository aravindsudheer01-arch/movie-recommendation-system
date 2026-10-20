"""Inspect raw dataset quality and representative metadata formatting.

Run from the project folder: python inspect_dataset.py
No dataset rows are uploaded or modified.
"""

import pandas as pd

from data_utils import DEFAULT_DATA_PATH, REQUIRED_COLUMNS, TEXT_FEATURES, load_movies

if not DEFAULT_DATA_PATH.is_file():
    raise SystemExit(f"Missing {DEFAULT_DATA_PATH}; copy your prepared data/movies.csv first.")

raw = pd.read_csv(DEFAULT_DATA_PATH, low_memory=False)
raw.columns = raw.columns.str.strip()
print(f"Raw dataset: {len(raw):,} rows, {len(raw.columns)} columns")
print("Missing required columns:", sorted(REQUIRED_COLUMNS - set(raw.columns)) or "none")
for feature in ("title", *TEXT_FEATURES):
    if feature in raw.columns:
        blank_count = raw[feature].fillna("").astype(str).str.strip().eq("").sum()
        print(f"  {feature:<10} missing/blank: {blank_count:,}")

if REQUIRED_COLUMNS <= set(raw.columns):
    movies = load_movies(DEFAULT_DATA_PATH)
    print(f"Usable movies: {len(movies):,}")
    print(f"Duplicate titles (before display disambiguation): {movies['title'].duplicated(keep=False).sum():,} rows")
    print("Example raw values (up to 2 movies, first 130 characters):")
    for _, record in raw.head(2).iterrows():
        print("  Movie:", str(record.get("title", ""))[:80])
        for feature in ("genres", "keywords", "cast", "director"):
            value = str(record.get(feature, ""))
            print(f"    {feature:<9}: {value[:130]}")
