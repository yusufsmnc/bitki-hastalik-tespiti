"""FastAPI uç noktalarının (endpoint) testleri.

TestClient gerçek bir sunucu başlatmadan uygulamaya istek atar; uvicorn
çalıştırmaya gerek yok. Model yine conftest.py'deki sahte model.
"""

import io

from fastapi.testclient import TestClient
from PIL import Image

import main

client = TestClient(main.app)


def png_bayt(boyut=(300, 300)):
    """Bellekte sahte bir PNG üretip bayt olarak döndürür."""
    tampon = io.BytesIO()
    Image.new("RGB", boyut, (34, 139, 34)).save(tampon, format="PNG")
    tampon.seek(0)
    return tampon.read()


def test_health_calisiyor_der():
    cevap = client.get("/health")
    assert cevap.status_code == 200
    assert cevap.json() == {"durum": "calisiyor"}


def test_kok_adres_arayuzu_dondurur():
    cevap = client.get("/")
    assert cevap.status_code == 200
    assert "text/html" in cevap.headers["content-type"]


def test_cors_baska_adresten_istege_izin_verir():
    # Tarayıcının başka bir adresten gelen istekte gönderdiği "Origin" başlığı.
    cevap = client.get("/health", headers={"Origin": "http://ornek.com"})
    assert cevap.headers["access-control-allow-origin"] == "*"


def test_predict_gorunt_kabul_eder():
    cevap = client.post(
        "/predict",
        files={"file": ("yaprak.png", png_bayt(), "image/png")},
    )
    assert cevap.status_code == 200

    sonuc = cevap.json()
    assert sonuc["durum"] in ("basarili", "emin_degil")
    assert 0.0 <= sonuc["guven"] <= 100.0


def test_predict_jpeg_de_kabul_eder():
    tampon = io.BytesIO()
    Image.new("RGB", (400, 300), (120, 160, 60)).save(tampon, format="JPEG")
    tampon.seek(0)

    cevap = client.post(
        "/predict",
        files={"file": ("yaprak.jpg", tampon.read(), "image/jpeg")},
    )
    assert cevap.status_code == 200
    assert "guven" in cevap.json()


def test_predict_dosyasiz_istek_reddedilir():
    # FastAPI eksik zorunlu alanda 422 döner; bunu kendimiz yazmamıza gerek yok.
    cevap = client.post("/predict")
    assert cevap.status_code == 422


def test_predict_exif_yonunu_uygular(monkeypatch):
    # 300x200 yatay bir fotoğraf, EXIF'te "90 derece döndür" (Orientation=6)
    # etiketiyle. Telefonlar dik çekilen fotoğrafı böyle kaydeder.
    img = Image.new("RGB", (300, 200), (34, 139, 34))
    exif = img.getexif()
    exif[0x0112] = 6  # 0x0112 = Orientation etiketi
    tampon = io.BytesIO()
    img.save(tampon, format="JPEG", exif=exif)

    # Modele giden görüntüyü yakalamak için tahmin fonksiyonunu değiştiriyoruz.
    gorulen = {}

    def sahte_tahmin(gelen_img):
        gorulen["boyut"] = gelen_img.size
        return {"durum": "basarili", "hastalik": "x", "guven": 99.0}

    monkeypatch.setattr(main, "tahmin_et_goruntu", sahte_tahmin)

    cevap = client.post(
        "/predict",
        files={"file": ("dik.jpg", tampon.getvalue(), "image/jpeg")},
    )
    assert cevap.status_code == 200
    # Döndürme uygulandıysa en ve boy yer değiştirmiş olmalı.
    assert gorulen["boyut"] == (200, 300)


def test_predict_resim_olmayan_dosyayi_400_ile_reddeder():
    cevap = client.post(
        "/predict",
        files={"file": ("not.txt", b"bu bir resim degil", "text/plain")},
    )
    assert cevap.status_code == 400
    assert "resim" in cevap.json()["detail"]


def test_predict_yarim_kalmis_resmi_400_ile_reddeder():
    # Gerçek bir PNG'nin sadece başı: format tanınır ama veri eksik.
    cevap = client.post(
        "/predict",
        files={"file": ("yarim.png", png_bayt()[:100], "image/png")},
    )
    assert cevap.status_code == 400


def test_predict_cok_buyuk_dosyayi_413_ile_reddeder(monkeypatch):
    # Testte gerçekten 10 MB üretmemek için sınırı geçici olarak küçültüyoruz.
    monkeypatch.setattr(main, "MAKS_DOSYA_BOYUTU", 100)
    cevap = client.post(
        "/predict",
        files={"file": ("buyuk.png", png_bayt(), "image/png")},
    )
    assert cevap.status_code == 413
