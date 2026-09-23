# 🎬 Cineverse Hub — Movie Recommendation Backend & ML Engine

A machine learning-powered movie recommendation backend built with **FastAPI**, **Scikit-Learn**, **Pandas**, and **KMeans Clustering**. It provides personalized movie recommendations based on plot content, genres, languages, and metadata similarity.

---

## 📌 Project Architecture

```
pj work/
├── app.py                     # FastAPI application with REST endpoints & CORS configuration
├── main.py                    # Entry point running Uvicorn server (http://127.0.0.1:8000)
├── recommender.py             # Inference engine using Euclidean distance & KMeans cluster filtering
├── Movies.py                  # Pydantic schema for API request validation
├── clustering_model.pkl       # Serialized model bundle (KMeans model, feature matrix, preprocessed DataFrame)
├── final_preprocessed_movies.csv # Cleaned & enriched movie dataset (27,800+ films with metadata)
├── Final_pp.ipynb             # Data preprocessing notebook (filtering, feature engineering)
├── model.ipynb                # ML pipeline notebook (TF-IDF, scaling, KMeans clustering, evaluation)
├── requirements.txt           # Project dependencies
└── .gitignore                 # Excludes raw data, checkpoints, and environment files
```

---

## 🚀 Machine Learning Pipeline

1. **Preprocessing (`Final_pp.ipynb`)**:
   - Cleaned TMDB movie dataset to filter active, quality titles.
   - Merged overview, genres, and keywords into a consolidated `content` feature.
   - Extracted and normalized numerical statistics (popularity, runtime, vote count, budget, revenue).

2. **Feature Engineering & Clustering (`model.ipynb`)**:
   - **Text Vectorization**: `TfidfVectorizer` applied to movie content (stop words removed).
   - **Numerical Scaling**: `StandardScaler` applied to runtime, popularity, votes, and budget.
   - **One-Hot Encoding**: Handled genres and primary languages.
   - **KMeans Clustering**: Grouped films into optimal clusters to narrow search space.
   - **Distance Metric**: `euclidean_distances` computes ranked similarity within clusters.

---

## 🛠️ Installation & Setup

### 1. Clone & Navigate to Project
```bash
git clone <YOUR_REPOSITORY_URL>
cd "pj work"
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI Server
```bash
python main.py
```
Or directly using Uvicorn:
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
Server runs at `http://127.0.0.1:8000`.  
Interactive Swagger API documentation is available at `http://127.0.0.1:8000/docs`.

---

## 📡 API Endpoints

### 1. Root / Welcome
- **URL**: `GET /`
- **Response**:
```json
{
  "message": "Hello World"
}
```

### 2. Movie Recommendations
- **URL**: `POST /predict`
- **Request Body**:
```json
{
  "title": "The Dark Knight",
  "n": 10
}
```
- **Response**:
```json
{
  "movie": {
    "id": 155,
    "title": "The Dark Knight",
    "original_title": "The Dark Knight",
    "overview": "Batman raises the stakes in his war on crime...",
    "poster_path": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
    "release_date": "2008-07-16",
    "release_year": 2008,
    "vote_average": 8.512,
    "vote_count": 30619,
    "runtime": 152,
    "genres": "Drama, Action, Crime, Thriller",
    "original_language": "en",
    "popularity": 130.643,
    "budget": 185000000,
    "revenue": 1004558444
  },
  "cluster": 10,
  "recommendations": [
    {
      "id": 157336,
      "title": "Interstellar",
      "original_title": "Interstellar",
      "overview": "The adventures of a group of explorers who make use of a newly discovered wormhole...",
      "poster_path": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
      "release_date": "2014-11-05",
      "release_year": 2014,
      "vote_average": 8.417,
      "vote_count": 32571,
      "runtime": 169,
      "genres": "Adventure, Drama, Science Fiction",
      "original_language": "en",
      "popularity": 140.241,
      "budget": 165000000,
      "revenue": 701729206,
      "similarity": 85
    }
  ]
}
```

---

## 📦 Tech Stack
- **FastAPI**: Modern, fast web framework for building APIs.
- **Uvicorn**: Lightning-fast ASGI server implementation.
- **Scikit-Learn**: Machine learning algorithms (KMeans, TF-IDF, Euclidean Distance).
- **Pandas & NumPy**: Data processing, matrix transformations, and feature engineering.
- **Pydantic**: Data validation and type enforcement.
