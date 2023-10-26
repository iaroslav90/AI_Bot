import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware



class Dense_v2(nn.Module):
    def __init__(self):
        super(Dense_v2, self).__init__()

        self.hidden_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.hidden_layer(x)


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

minmax_model = Dense_v2()
minmax_model.load_state_dict(torch.load('models/minmax.pth', map_location=device))
minmax_model.eval()


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
    try:
        if item.name == 'direction':
            data = item.data
            x = torch.tensor([data], dtype=torch.float32).to(device)
            x = (x - x[0,-1]) / x.std()
            y = nn.Sigmoid()(direction_model(x)).item()
            if y > 0.5:
                return "buy"
            else:
                return "sell"

        elif item.name == 'grid':
            data = item.data
            x = torch.tensor([data[:240]], dtype=torch.float32).view(1, 4, 60).transpose(1, 2).to(device)
            d = (x.max() - x.min()) / 5
            xx = (x - x[0, -1, 3]) / d
            y = (minmax_model(xx)[0] * d).cpu().numpy()
            return "%f,%f" % (max(y[0], 0) + data[-1], min(y[1], 0) + data[-1])
    except:
        return "Failed"


@app.get("/")
async def root():
    return "connection is okay."
