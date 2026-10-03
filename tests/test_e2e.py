"""Uçtan uca (end-to-end) testler: gerçek sunucu, gerçek HTTP.

Diğer testlerden FARKI bu: test_main.py TestClient kullanıyor, o uygulamayı
doğrudan Python'dan çağırır -- ortada ne uvicorn vardır, ne soket, ne de
gerçek bir HTTP isteği. Hızlıdır ama bazı hataları GÖREMEZ:

  - uvicorn uygulamayı import ederken patlıyorsa,
  - model açılışta yüklenemiyorsa,
  - static/ klasörü yanlış yere bağlanmışsa,
  - sunucu belirtilen portta hiç dinlemiyorsa.

Burada uvicorn'u ayrı bir süreç (process) olarak gerçekten başlatıyoruz ve
çiftçinin telefonunun atacağı isteklerin aynısını ağ üzerinden atıyoruz.
Yani "python dosyası çalışıyor mu" değil, "uygulama ayağa kalkıyor mu"
sorusunu test ediyoruz.

Model yine sahte: conftest.py'nin kurduğu MODEL_PATH / CLASS_NAMES_PATH
ortam değişkenleri alt sürece de aktarılıyor (env=os.environ.copy()).
"""

import io
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from PIL import Image

# Bu dosyadaki TÜM testler "e2e" işaretini taşır. CI'da ayrı bir adım olarak
# çalıştırabilmek için: pytest -m e2e  /  pytest -m "not e2e"
pytestmark = pytest.mark.e2e

KOK = Path(__file__).resolve().parents[1]
BACKEND = KOK / "bitki-backend"

# Sunucunun açılması torch'u import etmeyi ve modeli yüklemeyi içerir.
# Yerelde ~5 sn, CI'ın yavaş makinesinde daha uzun sürebilir; cömert olalım.
BASLATMA_ZAMAN_ASIMI = 120  # saniye


def _bos_port():
    """İşletim sisteminden boşta bir port iste.

    Port 0'a bağlanmak "bana uygun bir tane ver" demektir. Sabit bir port
    (8000 gibi) seçseydik, kullanıcının zaten çalışan sunucusuyla çakışır
    ve test sebepsiz yere kırmızı yanardı.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gunlugu_oku(gunluk_yolu):
    try:
        return gunluk_yolu.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(sunucu günlüğü okunamadı)"


@pytest.fixture(scope="module")
def sunucu(tmp_path_factory):
    """Gerçek bir uvicorn sunucusu başlatır, testler bitince kapatır.

    scope="module": sunucu her test için değil, bu dosya için BİR KERE
    açılır. Model yüklemesi pahalı; her testte yeniden başlatmak testleri
    dakikalarca sürdürürdü.
    """
    port = _bos_port()
    gunluk_yolu = tmp_path_factory.mktemp("e2e-sunucu") / "sunucu.log"

    # Çıktıyı dosyaya yazıyoruz, boruya (PIPE) değil: boruyu okumazsak
    # tampon dolduğunda sunucu süreci kilitlenebilir. Dosyadan ise hata
    # anında rahatça okuyup test mesajına koyabiliriz.
    gunluk = open(gunluk_yolu, "w", encoding="utf-8")

    surec = subprocess.Popen(
        # sys.executable: testleri çalıştıran Python'un ta kendisi. Böylece
        # yerelde venv içindeki, CI'da da kurulumdaki Python kullanılır.
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=BACKEND,              # "main:app" ancak bu klasörden import edilebilir
        env=os.environ.copy(),    # sahte model yollarını alt sürece taşır
        stdout=gunluk,
        stderr=subprocess.STDOUT,
    )

    adres = f"http://127.0.0.1:{port}"

    try:
        _hazir_olana_kadar_bekle(surec, adres, gunluk_yolu)
        yield adres
    finally:
        # terminate: nazik kapatma isteği. Sunucu 10 sn içinde kapanmazsa
        # kill ile zorla kapatıyoruz, yoksa test bitse de süreç arkada kalır.
        surec.terminate()
        try:
            surec.wait(timeout=10)
        except subprocess.TimeoutExpired:
            surec.kill()
        gunluk.close()


def _hazir_olana_kadar_bekle(surec, adres, gunluk_yolu):
    """/health cevap verene kadar bekler; vermezse anlaşılır bir hata üretir."""
    bitis = time.monotonic() + BASLATMA_ZAMAN_ASIMI

    while time.monotonic() < bitis:
        # Süreç çoktan öldüyse beklemeye devam etmenin anlamı yok.
        if surec.poll() is not None:
            pytest.fail(
                f"Sunucu açılamadan kapandı (çıkış kodu {surec.returncode}).\n"
                f"--- sunucu çıktısı ---\n{_gunlugu_oku(gunluk_yolu)}"
            )
        try:
            if httpx.get(f"{adres}/health", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            # Henüz dinlemiyor; normal. Biraz bekleyip tekrar deneyeceğiz.
            pass
        time.sleep(0.5)

    pytest.fail(
        f"Sunucu {BASLATMA_ZAMAN_ASIMI} sn içinde ayağa kalkmadı.\n"
        f"--- sunucu çıktısı ---\n{_gunlugu_oku(gunluk_yolu)}"
    )


def jpeg_bayt(boyut=(400, 300), renk=(34, 139, 34)):
    """Bellekte sahte bir 'yaprak' fotoğrafı üretir."""
    tampon = io.BytesIO()
    Image.new("RGB", boyut, renk).save(tampon, format="JPEG")
    return tampon.getvalue()


# --- 1. Aşama: sunucu ayakta mı ------------------------------------------

def test_01_sunucu_ayaga_kalkti_ve_health_cevap_veriyor(sunucu):
    """Bu test geçtiyse: uvicorn açıldı, main.py import edildi, model yüklendi.

    Tek başına projenin en değerli testi -- çünkü bu zincirin herhangi bir
    halkası kopsa sunucu hiç cevap veremezdi.
    """
    cevap = httpx.get(f"{sunucu}/health", timeout=10)
    assert cevap.status_code == 200
    assert cevap.json() == {"durum": "calisiyor"}


# --- 2. Aşama: arayüz servis ediliyor mu ---------------------------------

def test_02_kok_adres_arayuzu_veriyor(sunucu):
    cevap = httpx.get(f"{sunucu}/", timeout=10)
    assert cevap.status_code == 200
    assert "text/html" in cevap.headers["content-type"]
    # Sayfanın gerçekten bizim arayüzümüz olduğunu doğrula (boş/yanlış
    # bir HTML de 200 dönebilirdi).
    assert 'id="foto-secici"' in cevap.text


def test_03_static_klasoru_dogru_baglanmis(sunucu):
    # main.py'deki app.mount("/static", ...) çalışıyor mu?
    cevap = httpx.get(f"{sunucu}/static/index.html", timeout=10)
    assert cevap.status_code == 200
    assert "<!DOCTYPE html>" in cevap.text.upper() or "<html" in cevap.text.lower()


def test_04_otomatik_api_dokumani_acilyor(sunucu):
    # FastAPI'nin ürettiği /docs sayfası: API'yi elle denemenin en kolay yolu.
    assert httpx.get(f"{sunucu}/docs", timeout=10).status_code == 200


# --- 3. Aşama: asıl iş -- fotoğraf gönder, tahmin al ---------------------

def test_05_fotograf_gonderip_gecerli_sonuc_alinir(sunucu):
    """Çiftçinin yaşadığı akışın tamamı: fotoğraf -> HTTP -> model -> cevap."""
    cevap = httpx.post(
        f"{sunucu}/predict",
        files={"file": ("yaprak.jpg", jpeg_bayt(), "image/jpeg")},
        timeout=60,
    )
    assert cevap.status_code == 200

    sonuc = cevap.json()

    # Sözleşme: durum her zaman bu ikisinden biri olmalı.
    assert sonuc["durum"] in ("basarili", "emin_degil")
    assert isinstance(sonuc["guven"], float)
    assert 0.0 <= sonuc["guven"] <= 100.0

    # Duruma göre hangi alanların bulunması GEREKTİĞİ de sözleşmenin parçası.
    if sonuc["durum"] == "basarili":
        assert sonuc["hastalik"]
        assert sonuc["hastalik_tr"]
        assert "mesaj" not in sonuc
    else:
        assert sonuc["mesaj"]
        assert sonuc["en_yakin_tahmin"]
        assert "hastalik" not in sonuc


def test_06_farkli_boyut_ve_formatlar_calisir(sunucu):
    # Telefon fotoğrafları her boyutta gelir; hiçbiri sunucuyu düşürmemeli.
    for boyut in [(50, 80), (224, 224), (1600, 1200)]:
        cevap = httpx.post(
            f"{sunucu}/predict",
            files={"file": ("yaprak.jpg", jpeg_bayt(boyut), "image/jpeg")},
            timeout=60,
        )
        assert cevap.status_code == 200, f"{boyut} boyutunda hata"
        assert cevap.json()["durum"] in ("basarili", "emin_degil")


def test_07_ardisik_istekler_sunucuyu_bozmuyor(sunucu):
    """Model uygulama başında BİR KERE yüklenir (CLAUDE.md kısıt 4).

    Arka arkaya istek atıp hepsinin aynı şekilde cevaplandığını görüyoruz;
    biri yüklemeyi bozsaydı sonrakiler patlardı.
    """
    for _ in range(3):
        cevap = httpx.post(
            f"{sunucu}/predict",
            files={"file": ("yaprak.jpg", jpeg_bayt(), "image/jpeg")},
            timeout=60,
        )
        assert cevap.status_code == 200


# --- 4. Aşama: hatalı girdilere doğru tepki ------------------------------

def test_08_resim_olmayan_dosya_400_doner(sunucu):
    cevap = httpx.post(
        f"{sunucu}/predict",
        files={"file": ("not.txt", b"bu bir resim degil", "text/plain")},
        timeout=30,
    )
    assert cevap.status_code == 400
    # Arayüz bu "detail" alanını kullanıcıya gösteriyor; boş olmamalı.
    assert cevap.json()["detail"]


def test_09_bozuk_resim_400_doner(sunucu):
    cevap = httpx.post(
        f"{sunucu}/predict",
        files={"file": ("yarim.jpg", jpeg_bayt()[:80], "image/jpeg")},
        timeout=30,
    )
    assert cevap.status_code == 400


def test_10_10mb_ustu_dosya_413_doner(sunucu):
    # Burada sınırı küçültme numarası yapamayız: sunucu ayrı bir süreçte,
    # monkeypatch oraya ulaşmaz. Bu yüzden gerçekten büyük bir gövde yolluyoruz.
    buyuk = b"x" * (10 * 1024 * 1024 + 1024)
    cevap = httpx.post(
        f"{sunucu}/predict",
        files={"file": ("buyuk.jpg", buyuk, "image/jpeg")},
        timeout=60,
    )
    assert cevap.status_code == 413


def test_11_dosyasiz_istek_422_doner(sunucu):
    assert httpx.post(f"{sunucu}/predict", timeout=30).status_code == 422


# --- 5. Aşama: tarayıcı tarafı gereksinimleri ---------------------------

def test_12_cors_basligi_geliyor(sunucu):
    """Arayüz başka bir adresten servis edilirse tarayıcı bu başlığı arar."""
    cevap = httpx.get(
        f"{sunucu}/health",
        headers={"Origin": "http://ornek.com"},
        timeout=10,
    )
    assert cevap.headers.get("access-control-allow-origin") == "*"
