"""Run `python -m pytest -q` from the project root."""

import pandas as pd
import pytest

from data_utils import load_movies
from recommender import BaselineRecommender
from recommender2 import WeightedRecommender


@pytest.fixture()
def movie_csv(tmp_path):
    path = tmp_path / "movies.csv"
    pd.DataFrame(
        [
            dict(title="Orbit", id=1, release_date="2001-01-01", genres="Science Fiction", keywords="spaceship cosmos", overview="space adventure galaxy", director="Alex Ray", cast="Sam Lee"),
            dict(title="Orbit", id=2, release_date="2019-05-05", genres="Science Fiction", keywords="cosmos stars", overview="galaxy adventure", director="Alex Ray", cast="Sam Lee"),
            dict(title="Sea", id=3, release_date="2002-02-02", genres="Documentary", keywords="ocean reef", overview="underwater animals", director="Sue Park", cast="Kim Lee"),
            dict(title="Blank", id=4, release_date="2003-03-03", genres="", keywords="", overview="", director="", cast=""),
            dict(title="Movie", id=5, release_date="2004-03-03", genres='[{"name": "Science Fiction"}]', keywords='[{"name": "space travel"}]', overview="stars adventure", director='[{"name":"Alex Ray"}]', cast='[{"name": "Sam Lee"}]'),
        ]
    ).to_csv(path, index=False)
    return path


def test_loader_and_disambiguation(movie_csv):
    movies = load_movies(movie_csv)
    assert len(movies) == 5
    assert movies.loc[0, "display_title"] == "Orbit (2001)"
    assert movies.loc[1, "display_title"] == "Orbit (2019)"
    assert movies.loc[0, "genres"] == "science_fiction"
    assert movies.loc[0, "director"] == "alex_ray"
    assert movies.loc[4, "genres"] == "science_fiction"
    assert movies.loc[4, "director"] == "alex_ray"


def test_baseline_recommendations(movie_csv):
    model = BaselineRecommender(load_movies(movie_csv))
    result = model.recommend(0, top_n=4)
    assert not result.empty
    assert 0 not in result.index
    assert all(result["similarity"] > 0)


def test_weighted_recommendations_and_no_match(movie_csv):
    model = WeightedRecommender(load_movies(movie_csv))
    result = model.recommend(0, top_n=2, weights={"genres": 50, "keywords": 50})
    assert len(result) == 2  # A structured Science Fiction movie now shares the same genre.
    assert result.iloc[0]["title"] == "Orbit"
    assert result.iloc[1]["title"] == "Movie"
    assert all(result["similarity"] > 0)
    assert model.recommend(3).empty


def test_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"title": ["Incomplete"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_movies(path)


def test_preprocessing_original_tmdb(tmp_path):
    import json
    from prepare_dataset import prepare_dataset

    movies_csv = tmp_path / "tmdb_5000_movies.csv"
    credits_csv = tmp_path / "tmdb_5000_credits.csv"
    pd.DataFrame([{
        "id": 100, "title": "Sky",
        "genres": json.dumps([{"id": 878, "name": "Science Fiction"}]),
        "keywords": json.dumps([{"id": 10, "name": "space travel"}]),
        "overview": "An adventure among the stars", "release_date": "2002-03-04",
    }]).to_csv(movies_csv, index=False)
    pd.DataFrame([{
        "movie_id": 100, "title": "Sky",
        "cast": json.dumps([{"name": "Jane Smith"}, {"name": "John Doe"}]),
        "crew": json.dumps([{"name": "Alex Ray", "job": "Director"}, {"name": "Other", "job": "Writer"}]),
    }]).to_csv(credits_csv, index=False)
    output = tmp_path / "movies.csv"
    assert prepare_dataset(movies_csv, credits_csv, output) == 1
    loaded = load_movies(output)
    assert loaded.loc[0, "genres"] == "science_fiction"
    assert loaded.loc[0, "cast"] == "jane_smith john_doe"
    assert loaded.loc[0, "director"] == "alex_ray"
    with pytest.raises(FileExistsError):
        prepare_dataset(movies_csv, credits_csv, output)


def test_same_first_name_is_not_same_director(movie_csv):
    movies = load_movies(movie_csv)
    assert movies.loc[0, "director"] == "alex_ray"
    assert movies.loc[2, "director"] == "sue_park"


def test_tv_movie_is_one_genre(tmp_path):
    path = tmp_path / "movies.csv"
    pd.DataFrame([{
        "title": "TV Story", "genres": "TV Movie Drama", "keywords": "",
        "overview": "", "director": "", "cast": "",
    }]).to_csv(path, index=False)
    assert load_movies(path).loc[0, "genres"] == "tv_movie Drama"


def test_director_names_are_whole_entities(tmp_path):
    path = tmp_path / "movies.csv"
    pd.DataFrame([
        dict(title="A", genres="", keywords="", overview="", director="James Cameron", cast=""),
        dict(title="B", genres="", keywords="", overview="", director="James Wan", cast=""),
        dict(title="C", genres="", keywords="", overview="", director="James Cameron", cast=""),
    ]).to_csv(path, index=False)
    movies = load_movies(path)
    assert movies.loc[0, "director"] == "james_cameron"
    model = WeightedRecommender(movies)
    result = model.recommend(0, weights={"director": 100}, top_n=3)
    assert result["title"].tolist() == ["C"]
