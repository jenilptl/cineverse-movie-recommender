from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from Movies import Movie
from recommender import recommend_movie

app = FastAPI(title="Cineverse Hub Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def index():
  return {'message': 'Hello World'}

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