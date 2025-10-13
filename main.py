from fastapi import FastAPI


app = FastAPI()

# Application Programming Interface
@app.get("/")
def read_root():
  return {"Hello": "World"}

@app.get("/users/{path}")
def read_root(path: int, query: str, page: int):
  return {"path": path, "query": query, "page": page}


from pydantic import BaseModel

class Item(BaseModel):
  name: str
  price: float

@app.post("/items/")
def create_item(item: Item):
  return item

