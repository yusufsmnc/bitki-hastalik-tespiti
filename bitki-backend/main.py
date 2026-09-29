from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pathlib import Path
import io

from predict import tahmin_et_goruntu

app = FastAPI(title="Bitki Hastalığı Tespiti API")

# CORS: tarayıcı, bir sayfanın BAŞKA bir adresteki API'ye istek atmasını
# varsayılan olarak engeller. Bu ayar hangi adreslere izin verildiğini söyler.
# GELİŞTİRME AYARI: şu an her adrese ("*") izin veriliyor.
# production'da kısıtla: allow_origins'e sadece kendi alan adını yaz.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Arayüz dosyaları (HTML/CSS/JS) bu klasörde durur. Yol main.py'ye göre
# çözülür; uygulama hangi klasörden başlatılırsa başlatılsın bulunur.
STATIC_KLASORU = Path(__file__).resolve().parent / "static"

# /static/... adresine gelen istekler static/ klasöründeki dosyayla cevaplanır.
app.mount("/static", StaticFiles(directory=STATIC_KLASORU), name="static")

# Telefon fotoğrafları genelde 2-8 MB. Üst sınır koymazsak dev bir dosya
# olduğu gibi belleğe okunur ve sunucuyu zorlar.
MAKS_DOSYA_BOYUTU = 10 * 1024 * 1024  # 10 MB

# Kök adres açılınca doğrudan arayüzü göster; kullanıcı /static/index.html
# yazmak zorunda kalmasın.
@app.get("/")
def ana_sayfa():
    return FileResponse(STATIC_KLASORU / "index.html")

@app.get("/health")
def health():
    return {"durum": "calisiyor"}

# "async def" değil düz "def": tahmin CPU'yu meşgul eden senkron bir iş.
# async içinde çalışsaydı tahmin bitene kadar sunucu başka hiçbir isteğe
# cevap veremezdi. Düz def'i FastAPI ayrı bir thread'de çalıştırır.
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # Sınırın 1 bayt fazlasını okumak yeterli: o kadar varsa dosya zaten büyük.
    icerik = file.file.read(MAKS_DOSYA_BOYUTU + 1)
    if len(icerik) > MAKS_DOSYA_BOYUTU:
        raise HTTPException(status_code=413, detail="Dosya çok büyük (en fazla 10 MB).")

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