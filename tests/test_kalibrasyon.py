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


def maskesiz_pay_logitleri(hedef_pay, izinli, T=None):
    """MASKEDEN ÖNCE (tüm sınıflar açık) T-ölçekli softmax'ta, seçilen ürünün
    payını tam olarak hedef_pay yapan logit listesi üretir.

    Ürün tutarlılık uyarısının baktığı sayı budur: pay URUN_ESIGI'nin
    altındaysa uyarı çıkar. logitler_kur maskeden SONRAKİ olasılığı kurar;
    bu ise maskeden ÖNCEKİ payı kurar, ikisi farklı şeyler.

    izinli sınıflara a, diğerlerine 0 logit verilir. k=len(izinli),
    n=toplam sınıf olmak üzere:
        pay = k*e^(a/T) / (k*e^(a/T) + (n-k))
    Buradan e^(a/T) = hedef*(n-k) / (k*(1-hedef)), yani a = T*ln(...).

    T varsayılanı predict.SICAKLIK; pay hesabında T'nin etkisini göstermek
    isteyen test farklı bir T verebilir.
    """
    if T is None:
        T = predict.SICAKLIK
    if not 0.0 < hedef_pay < 1.0:
        raise ValueError("hedef_pay 0 ile 1 arasında olmalı")

    n = len(predict.class_names)
    k = len(izinli)
    r = hedef_pay * (n - k) / (k * (1 - hedef_pay))
    a = T * math.log(r)

    logitler = [0.0] * n
    for i in izinli:
        logitler[i] = a
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


# --- İki eşikli karar kuralı -------------------------------------------

def test_ayni_guvende_healthy_takilir_hastalik_takilmaz(logit_ver):
    """Kuralın ASIL davranışı: aynı güven (0.75), farklı karar.

    0.75 genel eşiği (0.70) geçer ama sağlıklı eşiğini (0.80) geçmez.
    Gerekçe: hastalıklı yaprağa "sağlıklı" demek tedaviyi geciktirir.
    """
    izinli = urun_indeksleri("domates")

    logit_ver(logitler_kur(0.75, DOMATES_SAGLIKLI, izinli))
    saglikli = tahmin_et_goruntu(yaprak(), "domates")

    logit_ver(logitler_kur(0.75, DOMATES_HASTALIK, izinli))
    hastalik = tahmin_et_goruntu(yaprak(), "domates")

    assert saglikli["durum"] == "emin_degil"
    assert saglikli["en_yakin_tahmin"] == "Tomato_healthy"

    assert hastalik["durum"] == "basarili"
    assert hastalik["hastalik"] == "Tomato_Leaf_Mold"

    # İkisinin güveni gerçekten aynı mıydı? Yoksa test başka bir şeyi
    # ölçüyor olurdu.
    assert saglikli["guven"] == pytest.approx(hastalik["guven"], abs=0.1)


def test_healthy_yeterince_yuksek_guvende_basarili_olur(logit_ver):
    """Sağlıklı eşiği bir duvar değil, daha yüksek bir çubuk: 0.85 geçer."""
    logit_ver(logitler_kur(0.85, DOMATES_SAGLIKLI, urun_indeksleri("domates")))

    sonuc = tahmin_et_goruntu(yaprak(), "domates")

    assert sonuc["durum"] == "basarili"
    assert sonuc["hastalik"] == "Tomato_healthy"


def test_genel_esigin_altinda_hastalik_da_emin_degil_olur(logit_ver):
    logit_ver(logitler_kur(0.65, DOMATES_HASTALIK, urun_indeksleri("domates")))

    sonuc = tahmin_et_goruntu(yaprak(), "domates")

    assert sonuc["durum"] == "emin_degil"


def test_esik_her_urunde_ayni_sekilde_ayrisir(logit_ver):
    """Kural domatese özel değil; üç üründe de aynı çalışmalı."""
    for urun in predict.GECERLI_URUNLER:
        izinli = urun_indeksleri(urun)
        saglikli_idx = next(
            i for i in izinli if "healthy" in predict.class_names[i].lower()
        )
        hastalik_idx = next(
            i for i in izinli if "healthy" not in predict.class_names[i].lower()
        )

        logit_ver(logitler_kur(0.75, saglikli_idx, izinli))
        assert tahmin_et_goruntu(yaprak(), urun)["durum"] == "emin_degil", urun

        logit_ver(logitler_kur(0.75, hastalik_idx, izinli))
        assert tahmin_et_goruntu(yaprak(), urun)["durum"] == "basarili", urun


# --- Sıcaklık ölçekleme -------------------------------------------------

def test_sicaklik_sonrasi_olasiliklar_bire_toplanir(logit_ver):
    """Maskelenenler 0 olduğu için izinli adayların toplamı 1 olmalı.

    Toplam 1 değilse maske softmax'tan SONRA uygulanmış demektir ve
    güven skorları ölçülen değerlerle uyumsuz olur.
    """
    logit_ver(logitler_kur(0.80, BIBER_HASTALIK, urun_indeksleri("biber")))

    sonuc = tahmin_et_goruntu(yaprak(), "biber")
    toplam = sum(a["guven"] for a in sonuc["adaylar"])

    # Biberin iki sınıfının ikisi de aday listesinde, yani tüm olasılık
    # kütlesi burada. round(.., 1) yüzünden 0.1 tolerans veriyoruz.
    assert toplam == pytest.approx(100.0, abs=0.2)


def test_sicaklik_tahmin_sirasini_degistirmez(logit_ver):
    """T'ye bölmek monoton bir dönüşüm: sıralamayı koruması gerekir.

    Aynı logit'lerin T'li ve T'siz sıralamasını karşılaştırıyoruz.
    """
    izinli = urun_indeksleri("domates")
    logitler = [50.0] * len(predict.class_names)
    # İzinli sınıflara birbirinden AYRIK değerler ver ki sıralama net olsun.
    for sira, i in enumerate(izinli):
        logitler[i] = float(sira)

    t_siz = torch.softmax(torch.tensor([[logitler[i] for i in izinli]]), dim=1)
    t_li = torch.softmax(
        torch.tensor([[logitler[i] for i in izinli]]) / predict.SICAKLIK, dim=1
    )

    assert torch.argsort(t_siz, descending=True).tolist() == \
        torch.argsort(t_li, descending=True).tolist()

    # Fonksiyonun kendi çıktısında da en yüksek logit kazanmalı.
    logit_ver(logitler)
    sonuc = tahmin_et_goruntu(yaprak(), "domates")
    en_yuksek_logit_idx = max(izinli, key=lambda i: logitler[i])
    tahmin = sonuc.get("hastalik") or sonuc["en_yakin_tahmin"]
    assert tahmin == predict.class_names[en_yuksek_logit_idx]


def test_sicaklik_guveni_dusurur(logit_ver):
    """T > 1 olasılıkları düzleştirir; kalibrasyonun yönü bu.

    Aynı logit'lerle T=1 ve T=1.95 karşılaştırılıyor.
    """
    izinli = urun_indeksleri("domates")
    logitler = [50.0] * len(predict.class_names)
    for i in izinli:
        logitler[i] = 0.0
    logitler[DOMATES_HASTALIK] = 4.0

    logit_ver(logitler)
    kalibreli = tahmin_et_goruntu(yaprak(), "domates")["guven"]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(predict, "SICAKLIK", 1.0)
        logit_ver(logitler)
        kalibresiz = tahmin_et_goruntu(yaprak(), "domates")["guven"]

    assert kalibreli < kalibresiz


# --- Ürün tutarlılık uyarısı -------------------------------------------
# Yanlış ürün seçildiğinde model emin görünen yanlış bir cevap verir:
# olasılık her zaman izin verilen sınıflara dağıtılır. Önlem, MASKEDEN ÖNCE
# seçilen ürünün sınıflarına düşen payı ölçmek; pay URUN_ESIGI'nin altındaysa
# "urun_uyarisi": True döner. Uyarı sonucu ENGELLEMEZ, sadece soru sorar.

def test_olasilik_baska_urunde_toplaninca_uyari_cikar(logit_ver):
    """Tüm sınıflar açıkken kütle çoğunlukla BAŞKA ürüne düşüyorsa uyarı çıkar.

    Domatesin bir sınıfına ezici logit verip ürünü "biber" seçiyoruz:
    maskesiz payın neredeyse tamamı domatese gider, biberin payı eşiğin
    çok altında kalır. Yanlış ürün seçiminin tam senaryosu bu.
    """
    logitler = [0.0] * len(predict.class_names)
    logitler[DOMATES_HASTALIK] = 50.0
    logit_ver(logitler)

    sonuc = tahmin_et_goruntu(yaprak(), "biber")

    assert sonuc["urun_uyarisi"] is True


def test_olasilik_secilen_urunde_toplaninca_uyari_cikmaz(logit_ver):
    """Kütle seçilen ürünün sınıfında toplanıyorsa uyarı çıkmaz."""
    logitler = [0.0] * len(predict.class_names)
    logitler[BIBER_HASTALIK] = 50.0
    logit_ver(logitler)

    sonuc = tahmin_et_goruntu(yaprak(), "biber")

    assert sonuc["urun_uyarisi"] is False


def test_esigin_hemen_altinda_uyari_true(logit_ver):
    """Pay 0.19 (eşik 0.20'nin hemen altında) -> uyarı."""
    izinli = urun_indeksleri("domates")
    logit_ver(maskesiz_pay_logitleri(0.19, izinli))

    assert tahmin_et_goruntu(yaprak(), "domates")["urun_uyarisi"] is True


def test_esigin_hemen_ustunde_uyari_false(logit_ver):
    """Pay 0.21 (eşik 0.20'nin hemen üstünde) -> uyarı yok."""
    izinli = urun_indeksleri("domates")
    logit_ver(maskesiz_pay_logitleri(0.21, izinli))

    assert tahmin_et_goruntu(yaprak(), "domates")["urun_uyarisi"] is False


def test_t_pay_hesabinda_uygulanir(logit_ver):
    """Pay hesabında SICAKLIK gerçekten uygulanıyor mu?

    AYNI logitlerle T=1 ve T=1.95'te uyarının farklı çıktığını gösteriyoruz.
    T>1 dağılımı düzleştirir: 2 sınıflı domatesin payı uniform değere
    (2/6≈0.33) doğru çekilir. Logitleri öyle kuruyoruz ki T=1'de pay 0.15
    (<0.20, uyarı var), T=1.95'te ise eşiğin üstüne çıksın (uyarı yok).
    T atlanırsa bu iki sonuç aynı olurdu ve 0.20 eşiği anlamını yitirirdi.
    """
    izinli = urun_indeksleri("domates")
    logitler = maskesiz_pay_logitleri(0.15, izinli, T=1.0)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(predict, "SICAKLIK", 1.0)
        logit_ver(logitler)
        t_bir = tahmin_et_goruntu(yaprak(), "domates")["urun_uyarisi"]

    logit_ver(logitler)
    t_kalibreli = tahmin_et_goruntu(yaprak(), "domates")["urun_uyarisi"]

    assert t_bir is True           # T=1: pay 0.15 < 0.20
    assert t_kalibreli is False    # T=1.95: pay eşiğin üstüne çıktı


def test_uyari_durumu_ve_adaylari_degistirmez(logit_ver):
    """Uyarı yalnızca ek bir bayrak: tahmin, durum, güven ve adaylar aynen kalır.

    Maskeli tahmin yalnızca İZİNLİ sınıfların logitlerine bağlı; maskelenen
    (ürün dışı) sınıfların değeri tahmini etkilemez ama MASKESİZ payı -- yani
    uyarıyı -- belirler. Bu ayrımı kullanarak aynı tahmini iki kez, bir kez
    uyarılı bir kez uyarısız üretiyoruz ve sonucun geri kalanının birebir
    aynı kaldığını doğruluyoruz.
    """
    izinli = urun_indeksleri("domates")

    # maskeli_logit=50 -> ürün dışı sınıflar baskın -> maskesiz pay ~0 -> uyarı
    logit_ver(logitler_kur(0.90, DOMATES_HASTALIK, izinli, maskeli_logit=50.0))
    uyarili = tahmin_et_goruntu(yaprak(), "domates")

    # maskeli_logit=-50 -> ürün dışı sınıflar sönük -> maskesiz pay ~1 -> uyarı yok
    logit_ver(logitler_kur(0.90, DOMATES_HASTALIK, izinli, maskeli_logit=-50.0))
    uyarisiz = tahmin_et_goruntu(yaprak(), "domates")

    assert uyarili["urun_uyarisi"] is True
    assert uyarisiz["urun_uyarisi"] is False

    # Uyarı dışındaki her şey aynı: durum, tahmin, güven, adaylar.
    assert uyarili["durum"] == uyarisiz["durum"] == "basarili"
    assert uyarili["hastalik"] == uyarisiz["hastalik"]
    assert uyarili["guven"] == uyarisiz["guven"]
    assert [a["hastalik"] for a in uyarili["adaylar"]] == \
        [a["hastalik"] for a in uyarisiz["adaylar"]]


# --- Geçersiz ürün ------------------------------------------------------

@pytest.mark.parametrize("kotu_urun", ["elma", "", "Domates", " domates ", "tomato"])
def test_gecersiz_urun_hata_firlatir(kotu_urun):
    """Katı davranış: büyük harf ve boşluk da geçersiz.

    Tek istemci kendi arayüzümüz ve sabit değer gönderiyor; toleranslı
    olmak gerçek bir istemci hatasını gizler.
    """
    with pytest.raises(ValueError) as hata:
        tahmin_et_goruntu(yaprak(), kotu_urun)

    # Hata mesajı ne yapılacağını söylemeli, sadece "geçersiz" dememeli.
    for urun in predict.GECERLI_URUNLER:
        assert urun in str(hata.value)


def test_urun_parametresi_zorunlu():
    """urun'u atlamak sessizce varsayılana düşmemeli, hata vermeli."""
    with pytest.raises(TypeError):
        tahmin_et_goruntu(yaprak())
