"""Load and validate a locally prepared TMDB movie catalogue."""

import ast
import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "movies.csv"
TEXT_FEATURES = ("genres", "keywords", "overview", "director", "cast")
REQUIRED_COLUMNS = {"title", *TEXT_FEATURES}


def _entity_token(name: str) -> str:
    """Encode an entire person or category name as one alphanumeric TF-IDF token."""
    return re.sub(r"(?u)[^\w]+", "_", name.strip().casefold()).strip("_")


def _normalise_metadata(value: str, feature: str) -> str:
    """Normalise metadata for feature-aware TF-IDF tokenisation.

    For structured lists, preserve each full name as one token. The supplied
    merged CSV flattens cast names, so its cast field remains a bag of words;
    actor identity cannot be recovered reliably without original credits data.
    """
    value = str(value).strip()
    if not value:
        return ""
    if not value.startswith("["):
        if feature == "director":
            # This particular dataset has one named director per row.
            return _entity_token(value)
        if feature == "genres":
            # Known multiword categories in the TMDB genre vocabulary.
            value = re.sub(r"\bscience\s+fiction\b", "science_fiction", value, flags=re.I)
            value = re.sub(r"\btv\s+movie\b", "tv_movie", value, flags=re.I)
        return value

    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            return value

    if not isinstance(parsed, list):
        return value
    names = []
    for item in parsed:
        if isinstance(item, dict):
            item = item.get("name", "")
        if isinstance(item, str) and item.strip():
            # Treat a full person/genre/keyword name as one token.
            names.append(_entity_token(item))
    return " ".join(names)


def load_movies(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the merged metadata CSV, returning rows with stable positional IDs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset missing: {path}. Place your prepared CSV at data/movies.csv."
        )

    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(
            "Dataset is missing columns: " + ", ".join(missing)
            + ". This app needs a merged/prepared TMDB dataset, "
            "not the two original Kaggle CSVs by themselves."
        )

    data["title"] = data["title"].fillna("").astype(str).str.strip()
    data = data.loc[data["title"].ne("")].copy().reset_index(drop=True)
    if data.empty:
        raise ValueError("Dataset has no movies with valid titles.")

    for col in TEXT_FEATURES:
        data[col] = data[col].fillna("").astype(str).str.strip()
        if col != "overview":
            data[col] = data[col].map(lambda value: _normalise_metadata(value, col))

    # Disambiguate identically named titles with release year, then catalogue ID.
    year = (
        data["release_date"].fillna("").astype(str).str.extract(r"^(\d{4})")[0]
        if "release_date" in data.columns
        else pd.Series("", index=data.index)
    )
    year = year.fillna("")
    data["display_title"] = data["title"] + year.map(
        lambda y: f" ({y})" if y else ""
    )
    duplicated = data["display_title"].duplicated(keep=False)
    if duplicated.any():
        def disambiguate(idx: int) -> str:
            movie_id = data.at[idx, "id"] if "id" in data.columns else None
            identifier = (
                str(movie_id).removesuffix(".0")
                if pd.notna(movie_id) and str(movie_id).strip()
                else f"row {idx + 1}"
            )
            return f"{data.at[idx, 'display_title']} [ID: {identifier}]"
        data.loc[duplicated, "display_title"] = [
            disambiguate(i) for i in data.index[duplicated]
        ]

    # Last-resort stable suffix for unexpected repeated metadata IDs.
    if data["display_title"].duplicated().any():
        for idx in data.index[data["display_title"].duplicated(keep=False)]:
            data.at[idx, "display_title"] += f" [row {idx + 1}]"

    return data
