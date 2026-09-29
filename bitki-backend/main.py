from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from predict import tahmin_et_goruntu

app = FastAPI(title="Bitki Hastalığı Tespiti API")

@app.get("/health")
def health():
    return {"durum": "calisiyor"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    icerik = await file.read()
    img = Image.open(io.BytesIO(icerik)).convert("RGB")
    sonuc = tahmin_et_goruntu(img)
    return sonuc