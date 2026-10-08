"""tahmin_et_goruntu fonksiyonunun sözleşmesini test eder.

Sahte model rastgele ağırlıklı olduğu için "doğru hastalığı buldu mu" diye
test EDEMEYİZ. Test ettiğimiz şey: fonksiyon her zaman beklenen biçimde
bir sonuç döndürüyor mu, ve güven eşiği mantığı doğru çalışıyor mu.

Eşiğin ve maskenin KESİN davranışı (belirli bir güven değerinde ne olacağı)
burada değil, test_kalibrasyon.py'de sabit logit'li bir stub ile test edilir.
"""

import pytest
from PIL import Image

import predict
from predict import tahmin_et_goruntu

# Testlerin çoğu için ürün önemli değil; tek bir yerde tanımlayıp
# her çağrıda tekrar yazmıyoruz.
URUN = "domates"


def yaprak_gorunt(boyut=(300, 300)):
    """Test için düz yeşil sahte bir 'yaprak' görüntüsü üretir."""
    return Image.new("RGB", boyut, (34, 139, 34))


@pytest.fixture
def her_zaman_basarili(monkeypatch):
    """İKİ eşiği de 0 yaparak sonucu her zaman 'basarili' yapar.

    Sadece GUVEN_ESIGI'ni sıfırlamak YETMEZ: tahmin bir "healthy" sınıfa
    denk gelirse SAGLIKLI_ESIGI devreye girer ve test rastgele ağırlıklara
    bağlı olarak bazen kırmızı yanardı (kararsız/flaky test).
    """
    monkeypatch.setattr(predict, "GUVEN_ESIGI", 0.0)
    monkeypatch.setattr(predict, "SAGLIKLI_ESIGI", 0.0)


@pytest.fixture
def her_zaman_emin_degil(monkeypatch):
    """İKİ eşiği de 1.01 yaparak sonucu her zaman 'emin_degil' yapar.

    1.01'i hiçbir olasılık geçemez (olasılık en fazla 1.0 olur).
    """
    monkeypatch.setattr(predict, "GUVEN_ESIGI", 1.01)
    monkeypatch.setattr(predict, "SAGLIKLI_ESIGI", 1.01)


def test_sonuc_sozluk_dondurur():
    sonuc = tahmin_et_goruntu(yaprak_gorunt(), URUN)
    assert isinstance(sonuc, dict)


def test_durum_alani_beklenen_degerlerden_biri():
    sonuc = tahmin_et_goruntu(yaprak_gorunt(), URUN)
    assert sonuc["durum"] in ("basarili", "emin_degil")


def test_guven_alani_her_zaman_var_ve_0_100_arasi():
    sonuc = tahmin_et_goruntu(yaprak_gorunt(), URUN)
    assert "guven" in sonuc
    assert isinstance(sonuc["guven"], float)
    assert 0.0 <= sonuc["guven"] <= 100.0


def test_urun_alani_gonderilenle_ayni_doner():
    # Arayüz hangi ürün için cevap geldiğini bilmek ister; backend
    # gönderileni aynen geri yansıtır.
    for urun in predict.GECERLI_URUNLER:
        sonuc = tahmin_et_goruntu(yaprak_gorunt(), urun)
        assert sonuc["urun"] == urun


def test_adaylar_her_iki_durumda_da_doner(her_zaman_basarili):
    # "basarili" durumunda da adaylar dönmeli (ürün kararı).
    sonuc = tahmin_et_goruntu(yaprak_gorunt(), URUN)
    assert sonuc["durum"] == "basarili"
    assert isinstance(sonuc["adaylar"], list)
    assert sonuc["adaylar"]


def test_aday_ogeleri_mevcut_adlandirmayi_izler():
    sonuc = tahmin_et_goruntu(yaprak_gorunt(), URUN)
    for aday in sonuc["adaylar"]:
        assert set(aday) == {"hastalik", "hastalik_tr", "guven"}
        assert aday["hastalik"] in predict.class_names
        assert aday["hastalik_tr"] == predict.turkce_ad(aday["hastalik"])
        # Birim mevcut "guven" alanıyla aynı: yüzde, 1 ondalık.
        assert 0.0 <= aday["guven"] <= 100.0
        assert round(aday["guven"], 1) == aday["guven"]


def test_esik_altinda_emin_degil_doner(her_zaman_emin_degil):
    sonuc = tahmin_et_goruntu(yaprak_gorunt(), URUN)

    assert sonuc["durum"] == "emin_degil"
    assert "mesaj" in sonuc
    assert sonuc["en_yakin_tahmin"] in predict.class_names
    assert sonuc["en_yakin_tahmin_tr"] == predict.turkce_ad(sonuc["en_yakin_tahmin"])
    assert "hastalik" not in sonuc


def test_esik_ustunde_basarili_doner(her_zaman_basarili):
    sonuc = tahmin_et_goruntu(yaprak_gorunt(), URUN)

    assert sonuc["durum"] == "basarili"
    assert sonuc["hastalik"] in predict.class_names
    assert sonuc["hastalik_tr"] == predict.turkce_ad(sonuc["hastalik"])
    assert "mesaj" not in sonuc


def test_farkli_boyuttaki_goruntuler_calisir():
    # Telefondan gelen fotoğraflar her boyutta olabilir; transform 224x224'e
    # indirdiği için hepsi sorunsuz geçmeli.
    for boyut in [(50, 80), (224, 224), (1200, 900)]:
        sonuc = tahmin_et_goruntu(yaprak_gorunt(boyut), URUN)
        assert sonuc["durum"] in ("basarili", "emin_degil")


def test_on_isleme_egitimdekiyle_ayni_kalmali():
    # Bu en sık yapılan deploy hatası: tahmin ön işlemesi eğitimdekinden
    # farklı olursa model saçmalar. Değerler yanlışlıkla değişirse bu test
    # kırmızı yanar.
    adimlar = predict.transform.transforms

    resize = adimlar[0]
    assert tuple(resize.size) == (224, 224)

    normalize = adimlar[-1]
    assert list(normalize.mean) == [0.485, 0.456, 0.406]
    assert list(normalize.std) == [0.229, 0.224, 0.225]


def test_bilinen_sinifin_turkce_adi_var():
    assert predict.turkce_ad("Tomato_Leaf_Mold") == "Domates - Yaprak küfü"


def test_bilinmeyen_sinif_ham_adiyla_doner():
    # Sözlükte olmayan bir sınıf gelirse çökmemeli, ham adı dönmeli.
    assert predict.turkce_ad("Elma_Kara_Leke") == "Elma_Kara_Leke"


def test_urun_haritasi_class_names_ile_tutarli():
    # Her sınıf tam bir ürüne ait olmalı; hiçbiri boşta kalmamalı, hiçbiri
    # iki ürüne sayılmamalı. (Önek eşlemesi bozulursa maske yanlış keser.)
    tum_indeksler = [i for liste in predict.URUN_INDEKSLERI.values() for i in liste]
    assert sorted(tum_indeksler) == list(range(len(predict.class_names)))
