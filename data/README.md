# Dataset setup

The application expects `movies.csv` in this folder.

Required columns: `title`, `genres`, `keywords`, `overview`, `director`, `cast`.
Optional columns: `release_date` and `id` (used to distinguish duplicate titles).

The file used during development appears to be a *prepared/merged* derivative
of the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
Kaggle's original distribution has separate movie and credits CSV files, so
simply renaming either source CSV to `movies.csv` will **not** work.

Place your existing prepared `movies.csv` here. It is excluded from Git until
its provenance and redistribution terms have been verified.

Alternatively, download the **two original CSV files** from Kaggle, place them
in this `data/` folder as `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`,
then run from the project root:

```bash
python prepare_dataset.py
```

This produces a new merged `movies.csv`, preserving metadata as named lists,
including up to five top-billed cast members per film by default. It will not
overwrite an existing `movies.csv` unless you explicitly pass `--overwrite`.
This prepared output may produce **different rankings** from a separately
preprocessed CSV, so choose one dataset consistently when comparing models.
