from fastapi import FastAPI
import json
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from gwp_anim import generate_results
from pydantic import BaseModel

class Product(BaseModel):
    premium: float
    loss_ratio: float
    term: int
    pattern_payment: int
    pattern_reporting: int
    comm_rate: float

class Model(BaseModel):
    model_idx: int

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['GET', 'POST']
)

prod1 = {
    'premium': 100,
    'loss_ratio': 0.7,
    'term': 12,
    'pattern_payment': 18,
    'pattern_reporting': 6,
    'comm_rate': 0.10
}

# with open('0.json') as f:
    # data = json.load(f)

@app.get("/")
async def root(product: Product, model: Model):
    return generate_results(product.dict(), model.dict())

@app.post("/")
async def root(product: Product, model: Model):
    return generate_results(product.dict(), model.dict())