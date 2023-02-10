import json

import joblib

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load('car_subscribe.pkl')
print(model['metadata'])

class Form(BaseModel):
    Client_ID: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    device_category: str
    device_brand: str
    device_browser_short: str
    diagonal: int
    geo_city: str
    time_action: int
    weekday: str
    month: str


class Prediction(BaseModel):
    Client_ID: str
    Result: float


@app.get('/status')
def status():
    return "service is available"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'Client_ID': form.Client_ID,
        'Result': y[0]
    }