"""tahmin_et_goruntu fonksiyonunun sözleşmesini test eder.

Sahte model rastgele ağırlıklı olduğu için "doğru hastalığı buldu mu" diye
test EDEMEYİZ. Test ettiğimiz şey: fonksiyon her zaman beklenen biçimde
bir sonuç döndürüyor mu, ve güven eşiği mantığı doğru çalışıyor mu.
"""

from PIL import Image

import predict
from predict import tahmin_et_goruntu


def yaprak_gorunt(boyut=(300, 300)):
    """Test için düz yeşil sahte bir 'yaprak' görüntüsü üretir."""
    return Image.new("RGB", boyut, (34, 139, 34))


def test_sonuc_sozluk_dondurur():
    sonuc = tahmin_et_goruntu(yaprak_gorunt())
    assert isinstance(sonuc, dict)


def test_durum_alani_beklenen_degerlerden_biri():
    sonuc = tahmin_et_goruntu(yaprak_gorunt())
    assert sonuc["durum"] in ("basarili", "emin_degil")


def test_guven_alani_her_zaman_var_ve_0_100_arasi():
    sonuc = tahmin_et_goruntu(yaprak_gorunt())
    assert "guven" in sonuc
    assert isinstance(sonuc["guven"], float)
    assert 0.0 <= sonuc["guven"] <= 100.0


def test_esik_altinda_emin_degil_doner(monkeypatch):
    # Eşiği 1.01 yaparsak hiçbir güven skoru yetmez -> her zaman "emin_degil".
    monkeypatch.setattr(predict, "GUVEN_ESIGI", 1.01)

    sonuc = tahmin_et_goruntu(yaprak_gorunt())

    assert sonuc["durum"] == "emin_degil"
    assert "mesaj" in sonuc
    assert sonuc["en_yakin_tahmin"] in predict.class_names
    assert sonuc["en_yakin_tahmin_tr"] == predict.turkce_ad(sonuc["en_yakin_tahmin"])
    assert "hastalik" not in sonuc


def test_esik_ustunde_basarili_doner(monkeypatch):
    # Eşiği 0.0 yaparsak her güven skoru yeter -> her zaman "basarili".
    monkeypatch.setattr(predict, "GUVEN_ESIGI", 0.0)

    sonuc = tahmin_et_goruntu(yaprak_gorunt())

    assert sonuc["durum"] == "basarili"
    assert sonuc["hastalik"] in predict.class_names
    assert sonuc["hastalik_tr"] == predict.turkce_ad(sonuc["hastalik"])
    assert "mesaj" not in sonuc


def test_farkli_boyuttaki_goruntuler_calisir():
    # Telefondan gelen fotoğraflar her boyutta olabilir; transform 224x224'e
    # indirdiği için hepsi sorunsuz geçmeli.
    for boyut in [(50, 80), (224, 224), (1200, 900)]:
        sonuc = tahmin_et_goruntu(yaprak_gorunt(boyut))
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
