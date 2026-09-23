from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from Movies import Movie
from recommender import recommend_movie, search_catalog

app = FastAPI(title="Cineverse Hub Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cineversebox-dac3toka5-vegadjenil2006-4757s-projects.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def index():
  return {'message': 'Hello World'}

@app.get('/movies')
def get_movies(search: str = "", genre: str = "All genres", limit: int = 50, page: int = 1):
  return search_catalog(query=search, genre=genre, limit=limit, page=page)

@app.get('/{name}')
def get_name(name: str):
  return {'Welcome To Cineverse Hub':f'{name}'}

@app.post('/predict')
def predict_movies(data: Movie):
  title = data.title.strip()
  n = data.n

  if title == "":
    raise HTTPException(status_code=400,detail="Movie title cannot be empty")

  if n <= 0:
    raise HTTPException(status_code=400,detail="n must be greater than 0")
  
  if n > 50:
    n = 50

  result = recommend_movie(title,n)
  if result is None:
    raise HTTPException(status_code=404,detail=f"Movie '{title}' not found")

  return result
