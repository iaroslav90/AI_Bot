import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


class Dense_direction_v1(nn.Module):
    def __init__(self):
        super(Dense_direction_v1, self).__init__()

        self.hidden_layer = nn.Sequential(
            nn.Linear(60, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.hidden_layer(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

direction_model = Dense_direction_v1()
direction_model.load_state_dict(torch.load('models/direction.pth', map_location=device))
direction_model.eval()


class Item(BaseModel):
    name: str
    data: list = []


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/model")
async def root(item: Item):
    if item.name == 'direction':
        data = item.data
        x = torch.tensor([data], dtype=torch.float32).to(device)
        x = (x - x[0,-1]) / x.std()
        y = nn.Sigmoid()(direction_model(x)).item()
        if y > 0.5:
            return {"direction" : "buy"}
        else:
            return {"direction" : "sell"}

    elif item.name == 'grid':
        data = item.data


@app.post("/test")
async def root():
    return "connection is okay."
