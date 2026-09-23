import os
import pickle
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

# Path to the pre-trained clustering model bundle
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "clustering_model.pkl")

# Load model bundle once at startup
with open(MODEL_PATH, "rb") as f:
    model_bundle = pickle.load(f)

model = model_bundle["model"]
df = model_bundle["df"]
X = model_bundle["X"]

def recommend_movie(title: str, n: int = 10):
    """
    Recommend n movies similar to the given title based on KMeans clustering
    and Euclidean distance on feature vectors.
    """
    movie = df[df["title"].astype(str).str.lower() == title.strip().lower()]
    if movie.empty:
        return None

    movie_index = movie.index[0]
    movie_features = X[movie_index].reshape(1, -1)
    prediction = model.predict(movie_features)
    cluster = int(prediction[0])
    cluster_indices = df[df["cluster"] == cluster].index

    distances = euclidean_distances(movie_features, X[cluster_indices])[0]
    result = pd.DataFrame({"index": cluster_indices, "distance": distances})
    result = result[result["index"] != movie_index]
    result = result.sort_values("distance")
    result = result.head(n)

    # Columns expected by the frontend
    cols = [
        "id", "title", "original_title", "overview", "poster_path",
        "release_date", "release_year", "vote_average", "vote_count",
        "runtime", "genres", "original_language", "popularity", "budget", "revenue"
    ]
    available_cols = [c for c in cols if c in df.columns]
    rec_df = df.loc[result["index"], available_cols].copy()

    # Similarity match percentage badge (60% - 99%)
    rec_df["similarity"] = [int(round(max(60, min(99, 100 - (d * 5))))) for d in result["distance"]]

    # Ensure full TMDB poster URL
    if "poster_path" in rec_df.columns:
        rec_df["poster_path"] = rec_df["poster_path"].apply(
            lambda x: f"https://image.tmdb.org/t/p/w500{x}" if isinstance(x, str) and x.startswith("/") else (x if isinstance(x, str) else "")
        )

    recommendations = rec_df.fillna("").to_dict(orient="records")

    queried_movie = df.loc[movie_index, available_cols].to_dict()
    poster = queried_movie.get("poster_path")
    if isinstance(poster, str) and poster.startswith("/"):
        queried_movie["poster_path"] = f"https://image.tmdb.org/t/p/w500{poster}"

    return {
        "movie": queried_movie,
        "cluster": cluster,
        "recommendations": recommendations
    }
