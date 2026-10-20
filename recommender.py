"""Version 1: single TF-IDF vector from genre, keyword and plot text."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BaselineRecommender:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies
        combined = (
            movies["genres"] + " " + movies["keywords"] + " " + movies["overview"]
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        try:
            self.matrix = self.vectorizer.fit_transform(combined)
        except ValueError as exc:
            raise ValueError("Not enough usable genre, keyword or plot text for TF-IDF.") from exc

    def recommend(self, movie_index: int, top_n: int = 5) -> pd.DataFrame:
        if not 0 <= movie_index < len(self.movies):
            raise IndexError("Selected movie is outside the catalogue.")
        similarities = cosine_similarity(
            self.matrix[movie_index], self.matrix
        ).ravel()
        similarities[movie_index] = -1  # Don't recommend the selected item.
        # Only positive matches, with reproducible tie-breaking by row position.
        order = np.lexsort((np.arange(len(similarities)), -similarities))
        indices = [i for i in order if similarities[i] > 0][:top_n]
        result = self.movies.iloc[indices].copy()
        result["similarity"] = similarities[indices]
        return result
