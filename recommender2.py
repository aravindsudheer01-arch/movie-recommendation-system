"""Version 2: separate TF-IDF models with configurable feature weights."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_utils import TEXT_FEATURES

DEFAULT_WEIGHTS = {
    "genres": 25,
    "keywords": 25,
    "overview": 25,
    "director": 10,
    "cast": 15,
}


class WeightedRecommender:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies
        self.matrices = {}
        for feature in TEXT_FEATURES:
            vectorizer = TfidfVectorizer(
                stop_words=None if feature == "director" else "english"
            )
            try:
                self.matrices[feature] = vectorizer.fit_transform(movies[feature])
            except ValueError:
                # Entire feature is empty (or only stopwords); keep app usable.
                self.matrices[feature] = None
        if all(matrix is None for matrix in self.matrices.values()):
            raise ValueError("The dataset has no usable text to build recommendations.")

    def recommend(
        self,
        movie_index: int,
        top_n: int = 5,
        weights: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        if not 0 <= movie_index < len(self.movies):
            raise IndexError("Selected movie is outside the catalogue.")
        weights = DEFAULT_WEIGHTS if weights is None else weights
        if any(weights.get(feature, 0) < 0 for feature in TEXT_FEATURES):
            raise ValueError("Feature weights must be non-negative.")

        individual = {}
        usable_weights = {}
        for feature in TEXT_FEATURES:
            matrix = self.matrices[feature]
            if matrix is None or matrix[movie_index].nnz == 0:
                individual[feature] = np.zeros(len(self.movies), dtype=float)
                continue
            individual[feature] = cosine_similarity(
                matrix[movie_index], matrix
            ).ravel()
            usable_weights[feature] = float(weights.get(feature, 0))

        weight_total = sum(usable_weights.values())
        final = np.zeros(len(self.movies), dtype=float)
        if weight_total > 0:
            for feature, weight in usable_weights.items():
                final += (weight / weight_total) * individual[feature]
        final[movie_index] = -1
        order = np.lexsort((np.arange(len(final)), -final))
        indices = [i for i in order if final[i] > 0][:top_n]
        result = self.movies.iloc[indices].copy()
        result["similarity"] = final[indices]
        for feature, values in individual.items():
            result[f"{feature}_similarity"] = values[indices]
        return result
