"""
uvicorn main:app --reload --host=0.0.0.0 --port=8000
"""

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

from keras.models import load_model
import numpy as np
from PIL import Image

app = FastAPI()
mm = load_model('mnist_model.h5')


class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None


# url/
@app.get("/")
def index():
    return {"Hello": "World"}


# url/predict
@app.get("/predict")
def predict():
    img = Image.open("test2.png").convert("L")
    img = np.resize(img, (1, 784))
    img = ((np.array(img) / 255) - 1) * -1
    np.argmax(mm.predict(img))
    return {"result": "a"}


# url/items/2?q=content
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}
