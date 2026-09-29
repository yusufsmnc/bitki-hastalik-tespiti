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
