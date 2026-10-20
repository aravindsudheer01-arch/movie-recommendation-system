# Movie Recommendation System — Baseline vs Weighted TF-IDF

A content-based movie recommendation app built with **Python, scikit-learn, pandas and Streamlit**.
The Streamlit app lets users search for a movie and get recommendations from the weighted model. The repository documents two modelling approaches:

- **Version 1 (`recommender.py`):** single TF-IDF model trained on combined genres, keywords and plot descriptions.
- **Version 2 (`recommender2.py`):** separate TF-IDF models for genres, keywords, overview, director and cast. Similarity scores are combined using configurable weights (defaults: 25%, 25%, 25%, 10%, 15%).

Both models use cosine similarity to rank results. Version 2 is the version shown in the app; Version 1 is retained in the repository as a learning baseline. 

## Features

- Search using a simple searchable dropdown that filters titles as you type, with **no movie selected on startup**. Duplicate titles are distinguished by release year.
- Get recommendations from the weighted (Version 2) algorithm; Version 1 is documented in code and in the model comparison below.
- Choose 1–10 recommendations; optionally adjust feature weights in the Advanced settings.
- See the overall similarity score and the Version 2 feature-by-feature breakdown.
- Cache fitted models across Streamlit interactions for responsiveness.
- Validate required data columns and handle missing metadata.

## Setup

1. Create a Python 3.10+ environment and install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Use your existing **merged** `data/movies.csv`, or download the two original Kaggle TMDB CSV files into `data/` and run `python prepare_dataset.py` to produce the merged file. This does not overwrite existing data by default (details in [`data/README.md`](data/README.md)).
3. Inspect the catalogue quality and metadata formatting:

   ```bash
   python inspect_dataset.py
   ```

4. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

## Searching for movies

Click **Search for a movie** and start typing. Streamlit filters the options inside the dropdown as you type—there are no separate search-result buttons. Select a matching title, then click **Recommend movies**. Nothing is selected automatically at startup. When multiple movies share a title, release years help identify the intended film. Choosing another movie clears the previous results.


## Model comparison (development)

| | Version 1 — baseline | Version 2 — weighted (Streamlit app) |
| --- | --- | --- |
| Inputs | Genres, keywords, overview | Genres, keywords, overview, director, cast |
| Text representation | One combined TF-IDF representation | Separate TF-IDF representation per feature |
| Scoring | Cosine similarity on combined representation | Weighted average of per-feature cosine similarities |
| Default weighting | Implicit via TF-IDF | Genres 25%, keywords 25%, overview 25%, director 10%, cast 15% |

Both models are preserved in source code and can be evaluated offline. The public-facing Streamlit interface uses Version 2 only to stay simple. Different recommendations are not proof of higher accuracy; an evaluation with relevance labels would be needed.

## Dataset

The uploaded dataset was inspected: **4,803 movies / 24 original columns**. The required fields are present. It contains 28 blank genres, 412 blank keyword lists, 4 missing/blank synopses, 43 missing cast strings and 30 missing directors. Three titles have multiple distinct movies (`Batman`, `The Host`, and `Out of the Blue`); the app labels them using release year. See [`DATASET_AUDIT.md`](DATASET_AUDIT.md).


This project was developed using a local, merged/processed movie dataset apparently based on [TMDB 5000 Movie Dataset (Kaggle)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). The precise preprocessing steps and redistribution terms should be verified before sharing the CSV publicly. The local 22 MB file is excluded from Git by default. Required fields are `title`, `genres`, `keywords`, `overview`, `director`, `cast`; `release_date` and `id` improve title disambiguation.

## Implementation notes

- The app uses TF-IDF vectors and cosine similarity; the scores are **not** predicted ratings or probabilities of enjoying a movie.
- Version 2 normalises the chosen weights among the selected movie's usable features so missing information does not penalise that movie unnecessarily.
- Recommendations with zero similarity are filtered out rather than returned as arbitrary matches.
- Structured TMDB name lists (JSON or Python-list format) are converted into single tokens per name, e.g. `sam_worthington`. For the supplied flattened CSV, the app also combines plain-text director names into single tokens and keeps `Science Fiction` / `TV Movie` intact. Plain-text **cast names remain individual words**: they cannot be reliably reconstructed from the flattened text. For actor-accurate comparisons, regenerate the dataset from the original movie and credits CSVs using `prepare_dataset.py`.

## Limitations / next steps

- Metadata-only approach: no personalised viewing history, ratings, or user evaluation data.
- Feature weights are hand-selected rather than learned or tuned against labeled relevance judgments.
- This already-preprocessed CSV flattens multiword **cast names**. The input quality is sufficient for experimentation, but cast-based matches can be misleading; use the original TMDB credits JSON to preserve actor identities.
- Compare models on a fixed list of example films and solicit relevance feedback to evaluate recommendation quality.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `DATASET_AUDIT.md` | Audit of the supplied 4,803-row CSV and observed limitations |
| `data_utils.py` | Dataset loading/cleaning and title disambiguation |
| `inspect_dataset.py` | One-command dataset quality/format report |
| `prepare_dataset.py` | Optional preprocessing of Kaggle's original movies/credits CSVs |
| `recommender.py` | Baseline TF-IDF model |
| `recommender2.py` | Weighted TF-IDF model |
| `tests/test_recommenders.py` | Recommender and preparation tests with a small synthetic dataset |

