# Dataset audit — supplied merged TMDB movie file

This audit was performed on the uploaded `movies(1).csv` (used locally as `data/movies.csv`). The dataset was not changed or redistributed.

- Original rows: **4,803**; columns: **24**; unique movie IDs: **4,803**.
- Missing/blank core values: `title`: **0**; `genres`: **28**; `keywords`: **412**; `overview`: **4**; `cast`: **43**; `director`: **30**.
- Three duplicate title strings refer to **six different movies**: `Batman` (1966/1989), `The Host` (2006/2013), and `Out of the Blue` (1980/2006). These are **not** duplicate movie IDs.
- Genre strings are already flattened. `Science Fiction` appears in **530** rows, and `TV Movie` appears in **8**. These phrases are preserved as single feature tokens by the revised loader.
- Cast strings in the supplied file are already flattened into space-delimited person names, e.g. `Sam Worthington Zoe Saldana`. Because there is no reliable delimiter between actors, TF-IDF on these strings can yield false matches on a common first/last name. Use the two original TMDB Kaggle CSVs with `prepare_dataset.py` for more reliable full-name matching (e.g. `sam_worthington`).
- Director strings are each a single name in this source dataset; the loader treats each full director name as one feature token (e.g. `james_cameron`).
- Many `homepage` and `tagline` entries are absent, but neither column is required for this content-based recommender.
- Ratings/popularity are metadata only and are not currently used in the similarity score. Similarity is **not** a probability or predicted user preference.
- The original source is *likely* the TMDB 5000 Movie Dataset from Kaggle, but exact lineage and redistribution rights should be confirmed before posting the CSV on GitHub; the provided `.gitignore` excludes it.

## Validation

Run `python inspect_dataset.py`, `python -m pytest -q` and `python -m streamlit run app.py` after placing the dataset at `data/movies.csv`. There are no claimed precision/recall or accuracy improvements without relevance labels or user feedback.
