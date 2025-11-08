from manga_translator.pipeline.pipeline import Pipeline
from manga_translator.utils.common import convert_img_to_base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from bs4 import BeautifulSoup
import requests
from pydantic import BaseModel
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_prompt = None

pipe = Pipeline(provider="openrouter", model="google/gemini-2.5-flash", device="cuda", 
                user_prompt=user_prompt)


class TranslateRequest(BaseModel):
    url: str
    process_image: bool = False

@app.post("/translate/")
async def translate_image(request: TranslateRequest):
    url = request.url
    process_image = request.process_image
    all_imgs = []

    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    imgs = soup.find_all("img")

    for img in imgs:
        src = img.get("src")
        if src.startswith("http"):
            all_imgs.append(src.split("?")[0])
    results = await pipe.run_batch(image_paths=all_imgs, process_image=process_image)

    results_base64 = [convert_img_to_base64(img) for img in results if img is not None]
    return {"images": results_base64}