from fastapi import FastAPI
import cv2
import numpy as np
import base64
from pydantic import BaseModel
from typing import Dict, List, Union , Optional , Tuple
from PIL import Image
import io , uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from PIL import Image , ImageDraw
from lamarem import LaMaRemover



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Only allow requests from this origin
    allow_credentials=True,
    allow_methods=["*"],  # Only allow specific HTTP methods
    allow_headers=["*"],  # Only allow specific headers
)

class Img(BaseModel):
    img : str
    masked :str

def inpaining(image_path, mask_path):
    image_path = image_path
    mask_path = mask_path
    remover = LaMaRemover()
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.fromarray(image_path).convert("RGB")
    # image = Image.open(image_path).convert("RGB")
    if isinstance(mask_path, str):
        mask = Image.open(mask_path).convert("L")
    else:
        mask = Image.fromarray(mask_path).convert("L")
    # mask = Image.open(mask_path).convert("L")
    output_image = remover(image, mask)
    output_image = np.array(output_image)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_image.jpg", output_image)
    return output_image

@app.get("/")
def read_root():
    return {"its working ...."}

@app.post("/inpainting")
async def inpaint(payload: Img):
    image = payload.img
    mask = payload.masked
    image = base64.b64decode(image)
    mask = base64.b64decode(mask)
    image = Image.open(io.BytesIO(image))
    mask = Image.open(io.BytesIO(mask))
    image = np.array(image)
    mask = np.array(mask)
    output = inpaining(image, mask)
    output = Image.fromarray(output)
    output = output.convert("RGB")
    buffered = io.BytesIO()
    output.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return {"output": img_str.decode('utf-8')}


if __name__ == "__main__":
    uvicorn.run(app, port = 8888)