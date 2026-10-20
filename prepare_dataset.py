"""Prepare a reproducible merged CSV from Kaggle's two raw TMDB 5000 files.

Example:
  python prepare_dataset.py --movies data/tmdb_5000_movies.csv \
      --credits data/tmdb_5000_credits.csv --output data/movies.csv

Does not overwrite existing output unless --overwrite is supplied.
"""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

MOVIE_COLUMNS = {"id", "title", "genres", "keywords", "overview", "release_date"}
CREDIT_COLUMNS = {"movie_id", "cast", "crew"}


def extract_entries(raw, column: str):
    """Read TMDB JSON arrays, with support for Python-literal representations."""
    if pd.isna(raw) or str(raw).strip() == "":
        return []
    if isinstance(raw, list):
        return raw
    try:
        entries = json.loads(str(raw))
    except (ValueError, TypeError):
        try:
            entries = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError, TypeError) as exc:
            raise ValueError(f"Could not parse {column} as a list: {str(raw)[:80]}") from exc
    if not isinstance(entries, list):
        raise ValueError(f"Expected a list in {column}, got {type(entries).__name__}")
    return entries


def names(raw, column: str, limit: int | None = None) -> str:
    values = [item.get("name", "").strip() for item in extract_entries(raw, column) if isinstance(item, dict)]
    values = [v for v in values if v]
    return json.dumps(values[:limit] if limit else values, ensure_ascii=False)


def directors(raw) -> str:
    people = []
    for item in extract_entries(raw, "crew"):
        if isinstance(item, dict) and str(item.get("job", "")).lower() == "director":
            n = item.get("name", "").strip()
            if n:
                people.append(n)
    return json.dumps(people, ensure_ascii=False)


def prepare_dataset(movie_path: Path, credit_path: Path, output: Path, overwrite: bool = False, cast_limit: int = 5) -> int:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists. Use --overwrite only if you intend to replace it.")
    movies = pd.read_csv(movie_path, low_memory=False)
    credits = pd.read_csv(credit_path, low_memory=False)
    missing_m = MOVIE_COLUMNS - set(movies.columns)
    missing_c = CREDIT_COLUMNS - set(credits.columns)
    if missing_m or missing_c:
        raise ValueError(f"Missing original TMDB columns: movies={sorted(missing_m)}, credits={sorted(missing_c)}")
    if movies["id"].duplicated().any() or credits["movie_id"].duplicated().any():
        raise ValueError("Source movie IDs must be unique before merging.")

    # Restrict to the fields used by the app; do not accidentally keep a second title column.
    joined = movies[list(MOVIE_COLUMNS)].merge(
        credits[["movie_id", "cast", "crew"]],
        left_on="id", right_on="movie_id", how="left", validate="one_to_one",
    )
    joined["genres"] = joined["genres"].apply(lambda x: names(x, "genres"))
    joined["keywords"] = joined["keywords"].apply(lambda x: names(x, "keywords"))
    joined["cast"] = joined["cast"].apply(lambda x: names(x, "cast", cast_limit))
    joined["director"] = joined["crew"].apply(directors)
    joined["overview"] = joined["overview"].fillna("")
    joined = joined[["id", "title", "genres", "keywords", "overview", "director", "cast", "release_date"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output, index=False)
    return len(joined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movies", type=Path, default=Path("data/tmdb_5000_movies.csv"))
    parser.add_argument("--credits", type=Path, default=Path("data/tmdb_5000_credits.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/movies.csv"))
    parser.add_argument("--cast-limit", type=int, default=5, help="Number of top-billed cast members per film")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.cast_limit < 1:
        parser.error("--cast-limit must be a positive integer")
    try:
        count = prepare_dataset(args.movies, args.credits, args.output, args.overwrite, args.cast_limit)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Dataset preparation failed: {exc}\n")
    print(f"Prepared {count:,} movies: {args.output}")
