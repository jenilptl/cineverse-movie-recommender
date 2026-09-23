from pydantic import BaseModel

class Movie(BaseModel):
  title: str
  n: int = 10