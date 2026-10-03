"""Arayüz ile backend arasındaki sözleşmeyi test eder.

Projede iki ayrı dünya var: Python tarafı (predict.py bir sözlük döndürür)
ve tarayıcı tarafı (index.html içindeki JS o sözlüğün alanlarını okur).
Bunları birbirine bağlayan hiçbir şey YOK -- biri değişirse diğeri sessizce
bozulur ve kimse fark etmez. Sunucu 200 döner, testler yeşil yanar, ama
çiftçinin ekranında "undefined" yazar.

Bu dosya o boşluğu kapatır: backend'in GERÇEKTEN döndürdüğü alanları
çalıştırarak toplar ve arayüzün onları kullandığını doğrular.

Tarayıcı açmaya gerek yok; HTML'i metin olarak okuyup JS'in hangi alanlara
dokunduğuna bakıyoruz. Kaba ama ucuz ve bu hatayı yakalamaya yeter.
"""

import re
from pathlib import Path

import pytest
from PIL import Image

import predict
from predict import tahmin_et_goruntu

KOK = Path(__file__).resolve().parents[1]
INDEX = KOK / "bitki-backend" / "static" / "index.html"
HTML = INDEX.read_text(encoding="utf-8")

# JS'te sunucu cevabı "veri" değişkenine konuyor: veri.durum, veri.guven...
# Bu desenle arayüzün okuduğu alan adlarını çıkarıyoruz.
ARAYUZUN_OKUDUGU = set(re.findall(r"veri\.(\w+)", HTML))

# Backend'in gönderdiği ama arayüzün BİLEREK göstermediği alanlar.
# Gerekçe: sistem "bu yaprağı tanıyamadım" dedikten sonra bir tahmin
# fısıldarsa çiftçi onu cevap sanar. Eşiğin 0.95 seçilme sebebi de buydu --
# yanlış yönlendirmektense susmak.
BILEREK_GOSTERILMEYEN = {"en_yakin_tahmin", "en_yakin_tahmin_tr"}


def yaprak():
    return Image.new("RGB", (300, 300), (34, 139, 34))


@pytest.fixture
def basarili_cevap(monkeypatch):
    """Eşiği 0 yaparak backend'i 'basarili' cevabı üretmeye zorlar."""
    monkeypatch.setattr(predict, "GUVEN_ESIGI", 0.0)
    return tahmin_et_goruntu(yaprak())


@pytest.fixture
def emin_degil_cevap(monkeypatch):
    """Eşiği 1.01 yaparak backend'i 'emin_degil' cevabı üretmeye zorlar."""
    monkeypatch.setattr(predict, "GUVEN_ESIGI", 1.01)
    return tahmin_et_goruntu(yaprak())


# --- İstek yönü: arayüz -> backend --------------------------------------

def test_arayuz_dogru_adrese_gonderiyor():
    assert 'fetch("/predict"' in HTML, "Arayüz /predict adresine POST etmeli."


def test_dosya_alan_adi_iki_tarafta_ayni():
    """JS'teki form.append("file", ...) ile main.py'deki `file` parametresi
    birebir aynı olmalı. Biri değişirse FastAPI alanı bulamaz ve 422 döner."""
    assert 'form.append("file"' in HTML

    main_kaynak = (KOK / "bitki-backend" / "main.py").read_text(encoding="utf-8")
    assert "def predict(file:" in main_kaynak


# --- Cevap yönü: backend -> arayüz --------------------------------------

def test_arayuz_olmayan_bir_alani_okumuyor(basarili_cevap, emin_degil_cevap):
    """Arayüz backend'in hiç göndermediği bir alanı okuyorsa, ekranda
    'undefined' çıkar. Yazım hatalarını (hastalik_tr -> hastalik_TR) yakalar."""
    backend_alanlari = set(basarili_cevap) | set(emin_degil_cevap)
    hayali = ARAYUZUN_OKUDUGU - backend_alanlari
    assert not hayali, f"Arayüz backend'de olmayan alan(lar)ı okuyor: {hayali}"


def test_basarili_cevabinin_alanlari_arayuzde_karsilaniyor(basarili_cevap):
    eksik = set(basarili_cevap) - BILEREK_GOSTERILMEYEN - ARAYUZUN_OKUDUGU
    assert not eksik, (
        f"'basarili' cevabında arayüzün hiç kullanmadığı alan(lar) var: {eksik}. "
        "Ya arayüzde göster, ya da gerekçesiyle BILEREK_GOSTERILMEYEN'e ekle."
    )


def test_emin_degil_cevabinin_alanlari_arayuzde_karsilaniyor(emin_degil_cevap):
    eksik = set(emin_degil_cevap) - BILEREK_GOSTERILMEYEN - ARAYUZUN_OKUDUGU
    assert not eksik, (
        f"'emin_degil' cevabında arayüzün hiç kullanmadığı alan(lar) var: {eksik}. "
        "Ya arayüzde göster, ya da gerekçesiyle BILEREK_GOSTERILMEYEN'e ekle."
    )


def test_en_yakin_tahmin_arayuzde_gosterilmiyor():
    """Bu bir ürün kararı, kaza değil -- testle sabitliyoruz ki ileride
    biri 'faydalı olur' diye ekleyince durup düşünsün."""
    assert "veri.en_yakin_tahmin" not in HTML


def test_durum_degerleri_iki_tarafta_ayni(basarili_cevap, emin_degil_cevap):
    """Arayüz durumu metinle karşılaştırıyor: veri.durum === "basarili".
    Backend bu dizeyi değiştirirse arayüz hiçbir sonucu tanıyamaz."""
    assert basarili_cevap["durum"] == "basarili"
    assert emin_degil_cevap["durum"] == "emin_degil"
    assert '=== "basarili"' in HTML
    assert '=== "emin_degil"' in HTML


def test_hata_mesaji_alani_iki_tarafta_ayni():
    """Backend hataları HTTPException(detail=...) ile döner; FastAPI bunu
    gövdede "detail" alanı olarak gönderir. Arayüz de onu okur."""
    assert "govde.detail" in HTML

    main_kaynak = (KOK / "bitki-backend" / "main.py").read_text(encoding="utf-8")
    assert "detail=" in main_kaynak


# --- Arayüzün kendi tutarlılığı -----------------------------------------

def test_sonuc_kartlari_gizlenebiliyor():
    """.gizli sınıfı, display tanımlayan diğer kurallara karşı kazanmalı.

    Bu testin sebebi gerçek bir hata: .yukleniyor-kart { display: flex }
    kuralı .gizli'yle aynı ağırlıktaydı ve CSS'te sonra yazıldığı için
    kazanıyordu. Sonuç: "Analiz ediliyor" kartı hiç gizlenemiyordu.
    """
    assert ".gizli { display: none !important; }" in HTML


def test_her_durumun_bir_cikis_butonu_var():
    """Üç sonuç ekranının da kullanıcıyı akışa geri döndüren bir butonu
    olmalı; yoksa çiftçi ekranda kilitli kalır."""
    for buton_id in ("yeni-foto-basarili", "yeni-foto-emin-degil", "tekrar-dene"):
        assert f'id="{buton_id}"' in HTML
