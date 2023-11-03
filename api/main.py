from fastapi import FastAPI, File, UploadFile
import uvicorn
from starlette.responses import JSONResponse
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import requests
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import json

MODEL = tf.keras.models.load_model("../Trained Model/1")
CropPredictionModel = joblib.load("../Crop Prediction Model/CropPredictionModel.joblib")


CLASS_NAMES = ['Potato___Early_blight',
               'Potato___healthy', 'POTATO__Late_blight']


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your React app's origin
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    user_input: float

class Package(BaseModel):
    n: float
    p: float
    k: float
    temperature:float	
    humidity:float	
    ph:float	
    rainfall:float

cropNameDict = {
    0: 'apple',
    1: 'banana',
    2: 'blackgram',
    3: 'chickpea',
    4: 'coconut',
    5: 'coffee',
    6: 'cotton',
    7: 'grapes',
    8: 'jute',
    9: 'kidneybeans',
    10: 'lentil',
    11: 'maize',
    12: 'mango',
    13: 'mothbeans',
    14: 'mungbean',
    15: 'muskmelon',
    16: 'orange',
    17: 'papaya',
    18: 'pigeonpeas',
    19: 'pomegranate',
    20: 'rice',
    21: 'watermelon'
}

    

@app.get("/hello")
async def root():
    return {"message": "Hello World"}


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predictCrop")
async def predictCrop(package:Package):
    values = [package.n, package.p, package.k, package.temperature, package.humidity, package.ph,package.rainfall]
    reshaped_values = [values]
    prediction = CropPredictionModel.predict(reshaped_values)
    
    return {"prediction": cropNameDict[float(prediction)]}


@app.post("/predictCropDisease")
async def predict(
    file: UploadFile = File(...)
):
    try:
        image = read_file_as_image(await file.read())
        print("image : ", image)

        img_batch = np.expand_dims(image, 0)
        print("image_batch : ", img_batch)

        prediction = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        return{
            'class': predicted_class,
            'confidence': float(confidence)

        }

        return JSONResponse(content={"message": f"Image uploaded successfully {prediction}"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
