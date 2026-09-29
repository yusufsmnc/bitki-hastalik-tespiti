from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageOps
import io

from predict import tahmin_et_goruntu

app = FastAPI(title="Bitki Hastalığı Tespiti API")

@app.get("/health")
def health():
    return {"durum": "calisiyor"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    icerik = await file.read()
    img = Image.open(io.BytesIO(icerik))
    # Telefonlar fotoğrafı döndürmek yerine EXIF'e "bu resim dönük" etiketi
    # yazar. Etiketi uygulamazsak model yan yatmış bir yaprak görür.
    img = ImageOps.exif_transpose(img).convert("RGB")
    sonuc = tahmin_et_goruntu(img)
    return sonuc