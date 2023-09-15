import requests
from fastapi import FastAPI, File, UploadFile
import uvicorn
from starlette.responses import JSONResponse
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

endpoint = "http://localhost:8501/v1/models/krishiVue:predict"


CLASS_NAMES = ['Potato___Early_blight',
               'Potato___healthy', 'POTATO__Late_blight']


app = FastAPI()


@app.get("/hello")
async def root():
    return {"message": "Hello World"}


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }


    response = requests.post(endpoint, json=(json_data))
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8001)
