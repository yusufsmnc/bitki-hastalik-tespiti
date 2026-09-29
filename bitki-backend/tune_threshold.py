"""Güven eşiğini veriyle seçmek için ölçüm script'i.

Ne yapar:
  plantdoc_split/test/ altındaki etiketli görüntüleri modelden geçirir,
  her biri için (gerçek etiket, tahmin, güven) toplar ve farklı güven
  eşikleri için bir tablo basar.

Neden gerekli:
  predict.py'deki GUVEN_ESIGI şu an tahminî bir değer. Eşiği yükseltmek
  "emin" dediğimiz tahminlerin isabetini artırır ama daha çok görüntüyü
  "emin değil"e atar (kapsama kaybı). Bu iki şey arasındaki dengeyi
  gözle görmeden seçmek keyfi olur -- tablo tam bunun için.

Nasıl çalıştırılır (bitki-backend/venv aktifken):
  python bitki-backend/tune_threshold.py
"""

from pathlib import Path

import torch
from PIL import Image

# predict.py'deki MODELİN ve TRANSFORM'un ta kendisini kullanıyoruz.
# Ayrı bir model/ön işleme kurmak ölçümü gerçek davranıştan koparırdı.
from predict import BURASI, class_names, model, transform

TEST_KLASORU = BURASI / "plantdoc_split" / "test"
ESIKLER = [0.50, 0.60, 0.70, 0.80, 0.90]
GORUNTU_UZANTILARI = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def goruntuleri_topla():
    """test/ altındaki sınıf klasörlerini gezip (yol, gerçek_etiket) listesi üretir."""
    if not TEST_KLASORU.is_dir():
        raise SystemExit(f"Test klasörü bulunamadı: {TEST_KLASORU}")

    kayitlar = []
    tanimsiz_klasorler = []

    for klasor in sorted(TEST_KLASORU.iterdir()):
        if not klasor.is_dir():
            continue

        # Klasör adı gerçek etikettir; modelin bildiği sınıflarla birebir
        # eşleşmezse o klasörü ölçüme katamayız.
        if klasor.name not in class_names:
            tanimsiz_klasorler.append(klasor.name)
            continue

        for yol in sorted(klasor.iterdir()):
            if yol.suffix.lower() in GORUNTU_UZANTILARI:
                kayitlar.append((yol, klasor.name))

    return kayitlar, tanimsiz_klasorler


def tahminleri_hesapla(kayitlar):
    """Her görüntüyü modelden geçirip sonuçları toplar."""
    sonuclar = []
    toplam = len(kayitlar)

    for sira, (yol, gercek) in enumerate(kayitlar, start=1):
        img = Image.open(yol).convert("RGB")
        x = transform(img).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            guven, tahmin_idx = torch.max(probs, 1)

        tahmin = class_names[tahmin_idx.item()]
        sonuclar.append({
            "gercek": gercek,
            "tahmin": tahmin,
            "guven": guven.item(),
            "dogru": tahmin == gercek,
        })

        if sira % 50 == 0 or sira == toplam:
            print(f"  işlendi: {sira}/{toplam}", flush=True)

    return sonuclar


def tabloyu_bas(sonuclar):
    toplam = len(sonuclar)
    genel_dogru = sum(1 for s in sonuclar if s["dogru"])

    print()
    print("=" * 74)
    print(f"Toplam görüntü: {toplam}")
    print(f"Eşiksiz genel doğruluk: {genel_dogru}/{toplam} "
          f"(%{genel_dogru / toplam * 100:.1f})")
    print("=" * 74)
    print()
    print(f"{'Eşik':<7}{'Emin tahmin':<16}{'İsabet':<12}"
          f"{'Emin değil (kapsama kaybı)':<28}")
    print("-" * 74)

    for esik in ESIKLER:
        eminler = [s for s in sonuclar if s["guven"] >= esik]
        emin_sayisi = len(eminler)
        emin_degil = toplam - emin_sayisi

        if emin_sayisi == 0:
            isabet_metni = "—"
        else:
            dogru = sum(1 for s in eminler if s["dogru"])
            isabet_metni = f"%{dogru / emin_sayisi * 100:.1f}"

        emin_metni = f"{emin_sayisi}/{toplam} (%{emin_sayisi / toplam * 100:.1f})"
        kayip_metni = f"{emin_degil} (%{emin_degil / toplam * 100:.1f})"

        print(f"{esik:<7.2f}{emin_metni:<16}{isabet_metni:<12}{kayip_metni:<28}")

    print("-" * 74)
    print()
    print("Okuma notu: 'İsabet', eşiği geçen tahminlerin yüzde kaçının doğru")
    print("olduğudur. Eşik yükseldikçe isabet artar ama daha çok görüntü")
    print("'emin değil'e düşer. Seçim bu ikisi arasındaki dengedir.")


def main():
    kayitlar, tanimsiz = goruntuleri_topla()

    if tanimsiz:
        print("UYARI: class_names.json'da karşılığı olmayan klasörler atlandı:")
        for ad in tanimsiz:
            print(f"  - {ad}")
        print()

    # Test setinde hiç örneği olmayan sınıflar: ölçüm o sınıflar hakkında
    # bir şey söylemez, bunu bilerek okumak gerekir.
    olculen_siniflar = {etiket for _, etiket in kayitlar}
    eksik = [ad for ad in class_names if ad not in olculen_siniflar]
    if eksik:
        print("NOT: Test setinde örneği olmayan sınıflar (ölçüme dahil değil):")
        for ad in eksik:
            print(f"  - {ad}")
        print()

    print(f"{len(kayitlar)} görüntü modelden geçirilecek...")
    sonuclar = tahminleri_hesapla(kayitlar)
    tabloyu_bas(sonuclar)


if __name__ == "__main__":
    main()
