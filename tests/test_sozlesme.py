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
from predict import tahmin_et_goruntu, GECERLI_URUNLER

KOK = Path(__file__).resolve().parents[1]
INDEX = KOK / "bitki-backend" / "static" / "index.html"
HTML = INDEX.read_text(encoding="utf-8")

# JS'te sunucu cevabı "veri" değişkenine konuyor: veri.durum, veri.guven...
# Bu desenle arayüzün okuduğu alan adlarını çıkarıyoruz.
ARAYUZUN_OKUDUGU = set(re.findall(r"veri\.(\w+)", HTML))

# Backend'in gönderdiği ama arayüzün BİLEREK göstermediği alanlar.
#
# Gerekçe: sistem "bu yaprağı tanıyamadım" dedikten sonra TEK bir tahmin
# fısıldarsa çiftçi onu cevap sanar. Yerine "adaylar" listesi gösteriliyor:
# liste belirsizliği açıkça anlatıyor ve boşuna değil -- eşiğin altında
# doğru cevap %88.4 oranında adayların içinde (CLAUDE.md Bölüm 1).
BILEREK_GOSTERILMEYEN = {"en_yakin_tahmin", "en_yakin_tahmin_tr"}

# Fixture'larda kullanılan ürün. Sahte sınıf listesinde (conftest.py)
# domatesin iki sınıfı var: bir hastalık, bir "healthy".
URUN = "domates"


def yaprak():
    return Image.new("RGB", (300, 300), (34, 139, 34))


def _esikleri_ayarla(monkeypatch, deger):
    """İKİ eşiği birlikte değiştirir.

    Sadece GUVEN_ESIGI'ni değiştirmek yetmez: tahmin bir "healthy" sınıfa
    denk gelirse predict.py daha sıkı olan SAGLIKLI_ESIGI'ni kullanır
    (_esik fonksiyonu). Sahte modelin ağırlıkları rastgele olduğu için
    hangisine denk geleceğini bilmiyoruz; tek eşiği patch'lemek testi
    modelin keyfine bırakırdı.
    """
    monkeypatch.setattr(predict, "GUVEN_ESIGI", deger)
    monkeypatch.setattr(predict, "SAGLIKLI_ESIGI", deger)


@pytest.fixture
def basarili_cevap(monkeypatch):
    """Eşikleri 0 yaparak backend'i 'basarili' cevabı üretmeye zorlar."""
    _esikleri_ayarla(monkeypatch, 0.0)
    return tahmin_et_goruntu(yaprak(), URUN)


@pytest.fixture
def emin_degil_cevap(monkeypatch):
    """Eşikleri 1.01 yaparak backend'i 'emin_degil' cevabı üretmeye zorlar."""
    _esikleri_ayarla(monkeypatch, 1.01)
    return tahmin_et_goruntu(yaprak(), URUN)


# --- İstek yönü: arayüz -> backend --------------------------------------

def test_arayuz_dogru_adrese_gonderiyor():
    assert 'fetch("/predict"' in HTML, "Arayüz /predict adresine POST etmeli."


def test_dosya_alan_adi_iki_tarafta_ayni():
    """JS'teki form.append("file", ...) ile main.py'deki `file` parametresi
    birebir aynı olmalı. Biri değişirse FastAPI alanı bulamaz ve 422 döner."""
    assert 'form.append("file"' in HTML

    main_kaynak = (KOK / "bitki-backend" / "main.py").read_text(encoding="utf-8")
    assert "def predict(file:" in main_kaynak


def test_urun_alani_iki_tarafta_ayni():
    """`urun` ZORUNLU bir form alanı: arayüz göndermezse 422 alır.

    Alanın kendisi kalibrasyonun ön şartı -- sıcaklık ve eşikler yalnızca
    ürün maskesi uygulanmış çıktılar üzerinde ölçüldü (CLAUDE.md Kısıt 5).
    """
    assert 'form.append("urun"' in HTML

    main_kaynak = (KOK / "bitki-backend" / "main.py").read_text(encoding="utf-8")
    assert "urun: str = Form(...)" in main_kaynak


def test_arayuzdeki_urun_degerleri_backendle_ayni():
    """Düğmelerdeki data-urun değerleri backend'in kabul ettiklerle birebir.

    Biri 'Domates' diye büyük harfle yazılsa sunucu 400 döner ve arayüz
    hiçbir tahmin üretemez; hata ise ancak elle denemede görünürdü.
    """
    arayuzdeki = set(re.findall(r'data-urun="(\w+)"', HTML))
    assert arayuzdeki == set(GECERLI_URUNLER)


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


def test_secilen_urun_arayuzde_gosteriliyor(basarili_cevap, emin_degil_cevap):
    """`urun` BİLEREK gösterilen bir alan -- yukarıdaki genel testin
    kapsadığından ayrı olarak burada da sabitliyoruz.

    Gerekçe ölçülmüş bir risk: yanlış ürün seçimi maskeyi yanlış sınıflara
    kurar, model yine yüksek güven verir ve cevap emin görünür (elle
    denemede domates yaprağı patates olarak %90.1 güvenle "Geç yanıklık").
    Çiftçinin bunu fark edebileceği tek yer bu satır. Biri ileride
    "gereksiz" diye kaldırmak isterse bu test durdurup düşündürsün.
    """
    assert "urun" in basarili_cevap and "urun" in emin_degil_cevap
    assert "veri.urun" in HTML
    assert "Seçilen bitki" in HTML


def test_adaylar_arayuzde_gosteriliyor(emin_degil_cevap):
    """Eşik altında adaylar listelenmeli; cevap da onları taşımalı."""
    assert emin_degil_cevap["adaylar"]
    assert "veri.adaylar" in HTML
    assert "Yaprak şunlardan biri olabilir" in HTML


def test_urun_uyarisi_arayuzde_okunuyor(basarili_cevap, emin_degil_cevap):
    """urun_uyarisi backend'in HER iki cevapta gönderdiği bir alan; arayüz
    onu okuyup yanlış ürün uyarısını gösteriyor olmalı.

    Yanlış ürün seçimi sistemin en büyük riski (CLAUDE.md Bölüm 1): maske
    yanlış sınıfları keser, model yine emin görünür. Bu bayrak, fotoğrafın
    seçilen bitkiye benzemediğini kullanıcıya SORARAK yakalar; engellemez.
    """
    assert "urun_uyarisi" in basarili_cevap
    assert "urun_uyarisi" in emin_degil_cevap
    assert "veri.urun_uyarisi" in HTML
    # Uyarı metni sunucudan gelen ürün adını içeriyor; XSS'e kapalı olması
    # için textContent ile yazılmalı (innerHTML değil).
    assert "urunUyariMesaji.textContent" in HTML


def test_en_yakin_tahmin_arayuzde_gosterilmiyor():
    """Bu bir ürün kararı, kaza değil -- testle sabitliyoruz ki ileride
    biri 'faydalı olur' diye ekleyince durup düşünsün.

    Tek bir tahmin, sistem "tanıyamadım" dedikten sonra bile cevap gibi
    görünür. Aday listesi ise belirsizliği açıkça gösteriyor ve eşiğin
    altında doğru cevap %88.4 oranında o listenin içinde.
    """
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
    """Dört sonuç ekranının da kullanıcıyı akışa geri döndüren bir butonu
    olmalı; yoksa çiftçi ekranda kilitli kalır."""
    for buton_id in ("yeni-foto-basarili", "yeni-foto-emin-degil",
                     "tekrar-dene", "onay-yeni-foto"):
        assert f'id="{buton_id}"' in HTML


# --- Ürün değişince ne OLMAMALI -----------------------------------------
# Buradaki testler bir davranışın varlığını değil YOKLUĞUNU koruyor:
# ürün düğmesine basmak tek başına istek göndermemeli.

def _fonksiyon_govdesi(ad):
    """HTML içindeki JS'ten bir fonksiyonun gövdesini süslü parantez
    sayarak çıkarır.

    Kaba bir yöntem (gerçek bir JS ayrıştırıcısı değil) ama bu dosyanın
    tarzına uygun: ucuz ve aradığımız hatayı yakalamaya yetiyor. Metin
    içinde süslü parantez geçmediği sürece doğru çalışır.
    """
    basla = HTML.index(f"function {ad}(")
    i = HTML.index("{", basla)
    derinlik = 0

    for j in range(i, len(HTML)):
        if HTML[j] == "{":
            derinlik += 1
        elif HTML[j] == "}":
            derinlik -= 1
            if derinlik == 0:
                return HTML[i:j + 1]

    raise AssertionError(f"{ad} fonksiyonunun sonu bulunamadı.")


def test_urun_degismesi_istek_gondermiyor():
    """Ürün düğmesine basmak fotoğrafı KENDİLİĞİNDEN göndermemeli.

    Sebep bir ürün kararı: "yanlış bitki seçtim, düzeltiyorum" ile "bu
    bitkiyi bitirdim, diğerine geçiyorum" niyeti dışarıdan aynı görünür --
    ikisinde de bir ürün düğmesine basılır. Otomatik gönderim ikinci
    durumda eski fotoğrafı yeni bitki olarak yollar ve tam da kaçınmak
    istediğimiz hatayı üretir: yanlış ürünle, emin görünen yanlış cevap.
    Bu yüzden sistem tahmin etmiyor, soruyor.
    """
    for ad in ("urunSec", "urunDegistigindeEkraniDuzelt"):
        govde = _fonksiyon_govdesi(ad)
        assert "tahminGonder" not in govde, (
            f"{ad} tahmin isteği gönderiyor. Ürün değişikliği istek "
            "göndermemeli; kullanıcıya onay kartıyla sorulmalı."
        )
        assert "fetch(" not in govde, f"{ad} doğrudan fetch çağırıyor."


def test_urun_degisince_eski_sonuc_gizleniyor():
    """Eski sonuç ekranda kalırsa seçili bitkiyle çelişir.

    durumGoster() her çağrıda ÖNCE hepsini gizleyip sonra istenenleri
    açıyor; bu fonksiyonun çağrılması "eski kart gizlendi" demek.
    """
    govde = _fonksiyon_govdesi("urunDegistigindeEkraniDuzelt")
    assert "durumGoster(" in govde


def test_ayni_fotografi_gondermek_onay_butonuna_bagli():
    """Aynı fotoğrafı yeni bitkiyle göndermenin TEK yolu onay düğmesi."""
    assert 'id="onay-gonder"' in HTML
    assert "onayGonder.addEventListener" in HTML
    assert "tahminGonder(sonDosya)" in HTML
