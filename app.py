"""Simple movie-search UI powered by the weighted TF-IDF recommender.

The baseline model is kept in recommender.py for learning and documentation,
but is intentionally not exposed as an end-user option in Streamlit.
"""

import streamlit as st

from data_utils import DEFAULT_DATA_PATH, TEXT_FEATURES, load_movies
from recommender2 import DEFAULT_WEIGHTS, WeightedRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🍿", layout="wide")
st.title("🍿 Movie Recommendation System")
st.write("Pick a movie you enjoyed to discover similar films.")


@st.cache_resource(show_spinner="Loading movies and building the recommender…")
def get_recommender(dataset_path: str, dataset_mtime: float):
    # Changes to the CSV modification time invalidate Streamlit's cached model.
    del dataset_mtime
    movies = load_movies(dataset_path)
    return movies, WeightedRecommender(movies)


try:
    if not DEFAULT_DATA_PATH.is_file():
        st.error("Dataset missing. Place `movies.csv` in the `data/` folder.")
        st.stop()
    movies, recommender = get_recommender(
        str(DEFAULT_DATA_PATH), DEFAULT_DATA_PATH.stat().st_mtime
    )
except (OSError, ValueError) as exc:
    st.error(f"Could not load the movie dataset: {exc}")
    st.stop()

with st.sidebar:
    st.header("Recommendation settings")
    top_n = st.slider("Number of recommendations", 1, 10, 5)
    weights = DEFAULT_WEIGHTS.copy()
    with st.expander("Advanced: feature weights", expanded=False):
        st.caption("Adjust how much each feature influences the recommendations.")
        for feature in TEXT_FEATURES:
            weights[feature] = st.slider(
                feature.title(),
                min_value=0,
                max_value=100,
                value=DEFAULT_WEIGHTS[feature],
                step=5,
                help="The model normalises the weights automatically.",
            )
    st.caption(f"{len(movies):,} movies available")

# Streamlit selectbox includes built-in type-to-search suggestions.
# index=None ensures there is NO automatic selection when the app opens.
ordered_indices = sorted(
    range(len(movies)), key=lambda i: movies.at[i, "display_title"].casefold()
)
selected_index = st.selectbox(
    "Search for a movie",
    options=ordered_indices,
    index=None,
    placeholder="Type a movie title, e.g. Avatar or Batman…",
    format_func=lambda i: movies.at[i, "display_title"],
    key="selected_movie",
    help="Start typing to filter the dropdown; select a matching title.",
)

# Switching movies should never leave the previous movie's results on screen.
if st.session_state.get("previous_selected_movie") != selected_index:
    st.session_state["previous_selected_movie"] = selected_index
    st.session_state["show_recommendations"] = False

if selected_index is not None:
    movie = movies.iloc[selected_index]

    with st.expander("About this movie"):
        st.write(movie["overview"] or "No plot summary available.")

    if st.button("Recommend movies", type="primary"):
        if sum(weights.values()) == 0:
            st.warning("Increase at least one feature weight above zero.")
            st.session_state["show_recommendations"] = False
        else:
            st.session_state["show_recommendations"] = True

    if st.session_state.get("show_recommendations", False):
        if sum(weights.values()) == 0:
            st.warning("Increase at least one feature weight above zero.")
        else:
            results = recommender.recommend(
                selected_index, top_n=top_n, weights=weights
            )
            st.subheader(f"Because you liked {movie['title']}…")
            if results.empty:
                st.info("No similar movies found. Try a different title or weights.")
            else:
                for _, result in results.iterrows():
                    st.markdown(f"#### 🎬 {result['display_title']}")
                    st.caption(
                        f"Similarity: {result['similarity']:.1%} "
                        "(not a predicted rating)"
                    )
                    st.write(result["overview"] or "No plot summary available.")
                    with st.expander("Why this movie?"):
                        for feature in TEXT_FEATURES:
                            score = result[f"{feature}_similarity"]
                            st.write(f"{feature.title()}: {score:.1%}")
                    st.divider()

st.caption("Recommendations are based on movie metadata, not viewing history.")
