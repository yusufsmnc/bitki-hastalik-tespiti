"""Ürün maskesi, sıcaklık ölçekleme ve iki eşikli karar kuralının testleri.

NEDEN AYRI BİR DOSYA: diğer testler conftest.py'nin RASTGELE AĞIRLIKLI
sahte resnet18'ini kullanıyor. O modelle "güven tam 0.75 olsun" diyemeyiz --
ne çıkarsa o çıkar. Oysa burada test edeceğimiz şeylerin tamamı belirli bir
güven değerinde ne olacağıyla ilgili.

Çözüm: modeli tamamen devre dışı bırakıp yerine istediğimiz logit'leri
döndüren küçük bir stub koyuyoruz (SabitLogitModel). Böylece her test
kendi senaryosunu matematikle kurar, şansa bırakmaz.

Sahte sınıf listesi (conftest.py):
    0 Pepper__bell___Bacterial_spot    biber   - hastalık
    1 Pepper__bell___healthy           biber   - sağlıklı
    2 Potato___Early_blight            patates - hastalık
    3 Potato___healthy                 patates - sağlıklı
    4 Tomato_Leaf_Mold                 domates - hastalık
    5 Tomato_healthy                   domates - sağlıklı
"""

import math

import pytest
import torch
from PIL import Image

import predict
from predict import tahmin_et_goruntu

# conftest.py'deki sahte listedeki indeksler. Ürün başına 2 sınıf var,
# yani maskeden sonra her zaman 2 sınıf izinli kalır.
BIBER_HASTALIK, BIBER_SAGLIKLI = 0, 1
PATATES_HASTALIK, PATATES_SAGLIKLI = 2, 3
DOMATES_HASTALIK, DOMATES_SAGLIKLI = 4, 5


class SabitLogitModel:
    """Her çağrıda verilen logit'leri aynen döndüren sahte model.

    Gerçek modelin yerine monkeypatch ile takılır; girdi görüntüsünü
    tamamen yok sayar, böylece sonuç yalnızca testin kurduğu logit'lere
    bağlı olur.
    """

    def __init__(self, logitler):
        self.logitler = logitler

    def __call__(self, x):
        # predict.py (1, sınıf_sayısı) biçiminde bir tensör bekliyor.
        return torch.tensor([self.logitler], dtype=torch.float32)


def yaprak():
    """Stub model girdiyi yok saydığı için içeriği önemsiz bir görüntü."""
    return Image.new("RGB", (64, 64), (34, 139, 34))


def logitler_kur(hedef_olasilik, kazanan, izinli, maskeli_logit=50.0):
    """İstenen güven değerini üretecek logit listesi hesaplar.

    hedef_olasilik: predict.py'nin HESAPLAYACAĞI güven, yani maskeleme ve
        SICAKLIK'a bölme işlemlerinden SONRAKİ softmax olasılığı.
    kazanan: en yüksek olasılığı alması istenen sınıfın indeksi.
    izinli: maskeden SONRA hayatta kalan indeksler, yani seçilen ürüne ait
        sınıflar. Formüldeki k = len(izinli) -- TOPLAM sınıf sayısı DEĞİL.
        Sahte listede her ürün 2 sınıflı olduğu için k = 2'dir; k'yı 6
        sanmak hedeflenen olasılığı tutturmaz.
    maskeli_logit: ürün dışı sınıflara verilen (kasten çok yüksek) değer.
        Maskeleme çalışmıyorsa bu sınıflar kazanır ve test bunu yakalar.

    Matematik: kaybedenlere 0 logit verilir, kazanana z. softmax(z/T)
    hedef p'ye eşit olsun istiyoruz:
        p = e^(z/T) / (e^(z/T) + (k-1))   =>   z = T * ln(p(k-1)/(1-p))
    """
    k = len(izinli)
    if not 0.0 < hedef_olasilik < 1.0:
        raise ValueError("hedef_olasilik 0 ile 1 arasında olmalı")
    if kazanan not in izinli:
        raise ValueError("kazanan, izinli indeksler arasında olmalı")

    z = predict.SICAKLIK * math.log(hedef_olasilik * (k - 1) / (1 - hedef_olasilik))

    logitler = [maskeli_logit] * len(predict.class_names)
    for i in izinli:
        logitler[i] = 0.0
    logitler[kazanan] = z
    return logitler


@pytest.fixture
def logit_ver(monkeypatch):
    """Verilen logit'leri döndüren stub'ı gerçek modelin yerine takar."""
    def tak(logitler):
        monkeypatch.setattr(predict, "model", SabitLogitModel(logitler))
    return tak


def urun_indeksleri(urun):
    return predict.URUN_INDEKSLERI[urun]


# --- Ürün maskesi -------------------------------------------------------

def test_urun_disi_siniflar_ne_tahmin_ne_aday_olur(logit_ver):
    """Maske gerçekten kesiyor mu?

    Domates/patates sınıflarına 50 gibi ezici bir logit verip ürünü "biber"
    seçiyoruz. Maske çalışmıyorsa tahmin kesin bir domates/patates sınıfı
    olurdu. Bu, Top-1 %57.64 -> %69.77 kazancının mekanizması.
    """
    logit_ver(logitler_kur(0.90, BIBER_HASTALIK, urun_indeksleri("biber")))

    sonuc = tahmin_et_goruntu(yaprak(), "biber")

    assert sonuc["hastalik"] == "Pepper__bell___Bacterial_spot"
    adaylar = [a["hastalik"] for a in sonuc["adaylar"]]
    assert all(ad.startswith("Pepper") for ad in adaylar), adaylar


def test_her_urun_icin_adaylar_sadece_o_urune_ait(logit_ver):
    for urun, onek in predict.URUN_ONEKLERI.items():
        izinli = urun_indeksleri(urun)
        logit_ver(logitler_kur(0.90, izinli[0], izinli))

        sonuc = tahmin_et_goruntu(yaprak(), urun)

        for aday in sonuc["adaylar"]:
            assert aday["hastalik"].startswith(onek)


def test_biberde_aday_sayisi_iki(logit_ver):
    """Biberin 2 sınıfı var; 3. aday icat edilmemeli."""
    logit_ver(logitler_kur(0.90, BIBER_HASTALIK, urun_indeksleri("biber")))

    sonuc = tahmin_et_goruntu(yaprak(), "biber")

    assert len(sonuc["adaylar"]) == 2


def test_aday_sayisi_en_fazla_uc_ve_sinif_sayisiyla_sinirli(logit_ver):
    for urun in predict.GECERLI_URUNLER:
        izinli = urun_indeksleri(urun)
        logit_ver(logitler_kur(0.90, izinli[0], izinli))

        sonuc = tahmin_et_goruntu(yaprak(), urun)

        assert len(sonuc["adaylar"]) == min(3, len(izinli))


def test_adaylar_azalan_sirada_ve_ilki_tahmin(logit_ver):
    logit_ver(logitler_kur(0.90, DOMATES_SAGLIKLI, urun_indeksleri("domates")))

    sonuc = tahmin_et_goruntu(yaprak(), "domates")
    guvenler = [a["guven"] for a in sonuc["adaylar"]]

    assert guvenler == sorted(guvenler, reverse=True)
    assert sonuc["adaylar"][0]["hastalik"] == "Tomato_healthy"
