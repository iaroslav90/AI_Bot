from datetime import datetime
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ftplib import FTP
from io import BytesIO, StringIO



class Dense_v1(nn.Module):
    def __init__(self):
        super(Dense_v1, self).__init__()

        self.hidden_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.hidden_layer(x)


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

sl_model = Dense_v1()
sl_model.load_state_dict(torch.load('models/sl_model.pth', map_location=device))
sl_model.eval()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return "connection is okay."

class ModelItem(BaseModel):
    name: str
    data: list = []

@app.post("/model")
async def root(item: ModelItem):
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
            y = (minmax_model(xx)[0] * d).detach().cpu().numpy()
            return "%f,%f" % (max(y[0], 0) + data[-1], min(y[1], 0) + data[-1])

        elif item.name == 'sl':
            data = item.data
            x = torch.tensor([data[:240]], dtype=torch.float32).view(1, 4, 60).transpose(1, 2).to(device)
            d = (x.max() - x.min()) / 5
            xx = (x - x[0, -1, 3]) / d
            y = (sl_model(xx)[0] * d).detach().cpu().numpy()
            return "%f,%f" % (max(y[0], 0) + data[-1], min(y[1], 0) + data[-1])
        
    except:
        return "Failed"


with open('accounts.json', 'r') as f:
    accounts = json.load(f)

class LicenseItem(BaseModel):
    mail: str
    account: str


@app.post("/license")
async def root(item: LicenseItem):
    ftp = FTP('ftp.theauroraai.com') 
    ftp.login(user='license@theauroraai.com', passwd = 'J,MAR&_welCm')
    def grabFile(filename):
        r = BytesIO()
        ftp.retrbinary('RETR %s' % filename, r.write)
        return r.getvalue()

    csv_data = grabFile('clients.csv')
    data = StringIO(csv_data.decode('utf-8'))
    license = pd.read_csv(data)

    if len(license.index[license['Payment Email'] == item.mail].tolist()) == 0:
        return "false,not registered email,"

    idx = license.index[license['Payment Email'] == item.mail].tolist()[0]
    date = license['Date of Expiry'].loc[idx]

    if datetime.now().strftime("%Y.%m.%d") > license['Date of Expiry'].loc[idx]:
        return "false,license expired,"

    if item.mail not in accounts:
        accounts[item.mail] = []
    if item.account not in accounts[item.mail] and len(accounts[item.mail]) >= 5:
        return "false,this email is used for more than 5 accounts,"
    if item.account not in accounts[item.mail]:
        accounts[item.mail].append(item.account)
        try:
            with open('accounts.json', 'w') as f:
                json.dump(accounts, f, indent=4)
        except:
            pass
    
    return "ok,%s,%s" % (license['Date of Expiry'].loc[idx], len(accounts[item.mail]))