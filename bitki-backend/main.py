from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image, ImageOps
import io

from predict import tahmin_et_goruntu

app = FastAPI(title="Bitki Hastalığı Tespiti API")

@app.get("/health")
def health():
    return {"durum": "calisiyor"}

# "async def" değil düz "def": tahmin CPU'yu meşgul eden senkron bir iş.
# async içinde çalışsaydı tahmin bitene kadar sunucu başka hiçbir isteğe
# cevap veremezdi. Düz def'i FastAPI ayrı bir thread'de çalıştırır.
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    icerik = file.file.read()
    try:
        img = Image.open(io.BytesIO(icerik))
        # Telefonlar fotoğrafı döndürmek yerine EXIF'e "bu resim dönük" etiketi
        # yazar. Etiketi uygulamazsak model yan yatmış bir yaprak görür.
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (OSError, Image.DecompressionBombError):
        # OSError: resim değil ya da bozuk/yarım dosya.
        # DecompressionBombError: küçük dosya ama devasa piksel sayısı (saldırı).
        # İkisinde de sunucu çökmesin, kullanıcıya anlaşılır bir cevap dönsün.
        raise HTTPException(status_code=400, detail="Yüklenen dosya okunabilir bir resim değil.")
    sonuc = tahmin_et_goruntu(img)
    return sonuc