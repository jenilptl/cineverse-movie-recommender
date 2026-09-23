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

import unicodedata
import re

# Helper for accent and punctuation-insensitive search (e.g. Bāhubali -> bahubali, K.G.F -> kgf)
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8").lower()
    cleaned = re.sub(r"[^a-z0-9\s]", "", normalized)
    return re.sub(r"\s+", " ", cleaned).strip()

# Precompute normalized search columns once at startup
df["_norm_title"] = df["title"].apply(normalize_text)
df["_norm_orig"] = df["original_title"].apply(normalize_text) if "original_title" in df.columns else df["_norm_title"]

def recommend_movie(title: str, n: int = 10):
    """
    Recommend n movies similar to the given title based on KMeans clustering
    and Euclidean distance on feature vectors.
    """
    if not title or not title.strip():
        return None

    raw_query = title.strip().lower()
    norm_query = normalize_text(title)
    norm_query_alt = norm_query.replace("aa", "a")

    # 1. Exact match on raw title or original title
    movie = df[df["title"].astype(str).str.lower() == raw_query]
    if movie.empty and "original_title" in df.columns:
        movie = df[df["original_title"].astype(str).str.lower() == raw_query]

    # 2. Normalized exact match (handles accents like Bāhubali == Bahubali)
    if movie.empty:
        movie = df[(df["_norm_title"] == norm_query) | (df["_norm_orig"] == norm_query)]

    # 3. Normalized double-letter variation (handles Baahubali == Bahubali)
    if movie.empty:
        movie = df[(df["_norm_title"] == norm_query_alt) | (df["_norm_orig"] == norm_query_alt)]

    # 4. Normalized prefix / contains match sorted by popularity
    if movie.empty:
        matches = df[
            df["_norm_title"].str.contains(norm_query, regex=False) |
            df["_norm_title"].str.contains(norm_query_alt, regex=False) |
            df["_norm_orig"].str.contains(norm_query, regex=False)
        ]
        if not matches.empty:
            movie = matches.sort_values(by="popularity", ascending=False).head(1)

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


import math

def search_catalog(
    query: str = "",
    genre: str = "All genres",
    language: str = "All languages",
    min_rating: float = 0.0,
    year: str = "",
    sort_by: str = "id-asc",
    limit: int = 24,
    page: int = 1
):
    """
    Search, filter, sort, and paginate through the full 34,791 movies catalog.
    """
    filtered = df

    # 1. Search Query
    if query and str(query).strip():
        raw_q = str(query).strip().lower()
        norm_q = normalize_text(query)
        norm_q_alt = norm_q.replace("aa", "a")
        mask = (
            filtered["title"].astype(str).str.lower().str.contains(raw_q, regex=False) |
            filtered["_norm_title"].str.contains(norm_q, regex=False) |
            filtered["_norm_title"].str.contains(norm_q_alt, regex=False) |
            filtered["_norm_orig"].str.contains(norm_q, regex=False) |
            filtered["genres"].astype(str).str.lower().str.contains(raw_q, regex=False)
        )
        filtered = filtered[mask]

    # 2. Genre Filter
    if genre and genre != "All genres":
        filtered = filtered[filtered["genres"].astype(str).str.contains(genre, regex=False)]

    # 3. Language Filter
    if language and language != "All languages":
        filtered = filtered[filtered["original_language"].astype(str) == str(language).strip()]

    # 4. Minimum Rating Filter
    if min_rating and float(min_rating) > 0:
        filtered = filtered[filtered["vote_average"].astype(float) >= float(min_rating)]

    # 5. Release Year Filter
    if year and str(year).strip():
        y = str(year).strip()
        filtered = filtered[
            filtered["release_date"].astype(str).str.startswith(y) |
            (filtered["release_year"].astype(str) == y)
        ]

    # 6. Sorting (Default: id-asc, starting with ID from smallest to largest)
    if sort_by == "id-desc":
        filtered = filtered.sort_values(by="id", ascending=False)
    elif sort_by == "popularity-desc":
        filtered = filtered.sort_values(by="popularity", ascending=False)
    elif sort_by == "popularity-asc":
        filtered = filtered.sort_values(by="popularity", ascending=True)
    elif sort_by == "rating-desc":
        filtered = filtered.sort_values(by="vote_average", ascending=False)
    elif sort_by == "rating-asc":
        filtered = filtered.sort_values(by="vote_average", ascending=True)
    elif sort_by == "year-desc":
        filtered = filtered.sort_values(by="release_year", ascending=False)
    elif sort_by == "year-asc":
        filtered = filtered.sort_values(by="release_year", ascending=True)
    elif sort_by == "title-asc":
        filtered = filtered.sort_values(by="title", ascending=True)
    elif sort_by == "title-desc":
        filtered = filtered.sort_values(by="title", ascending=False)
    else:
        # Default: id-asc
        filtered = filtered.sort_values(by="id", ascending=True)

    total_count = len(filtered)
    page = max(1, int(page))
    limit = max(1, min(100, int(limit)))
    total_pages = max(1, math.ceil(total_count / limit))

    start = (page - 1) * limit
    end = start + limit
    paged = filtered.iloc[start:end]

    cols = [
        "id", "title", "original_title", "overview", "poster_path",
        "release_date", "release_year", "vote_average", "vote_count",
        "runtime", "genres", "original_language", "popularity", "budget", "revenue"
    ]
    available_cols = [c for c in cols if c in paged.columns]
    paged_df = paged[available_cols].copy()

    if "poster_path" in paged_df.columns:
        paged_df["poster_path"] = paged_df["poster_path"].apply(
            lambda x: f"https://image.tmdb.org/t/p/w500{x}" if isinstance(x, str) and x.startswith("/") else (x if isinstance(x, str) else "")
        )

    return {
        "total": total_count,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
        "movies": paged_df.fillna("").to_dict(orient="records")
    }
