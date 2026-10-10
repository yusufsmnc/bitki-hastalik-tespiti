import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Yollar bu dosyanın bulunduğu klasöre göre çözülür; böylece uygulama
# hangi klasörden başlatılırsa başlatılsın model dosyaları bulunur.
BURASI = Path(__file__).resolve().parent


# --- Yapılandırma -------------------------------------------------------
# Sıcaklık ve eşikler KODUN değil, MODELİN özelliği: üçü de Colab'da bu
# belirli ağırlık dosyası üzerinde ölçüldü. Yeni bir model eğitilirse üçü
# birlikte değişir. Bu yüzden model dosya adıyla aynı yerde, ayrı bir
# "model künyesi" dosyasında duruyorlar (class_names.json gibi).
YAPILANDIRMA_YOLU = BURASI / "model_config.json"


def _yapilandirmayi_oku(yol):
    """model_config.json'u okur ve beklenen alanların varlığını doğrular.

    Dosya bozuk veya eksikse açılışta hata verir: yarım kalibrasyonla
    tahmin üretmek, hiç açılmamaktan daha kötü.
    """
    try:
        with open(yol, encoding="utf-8") as f:
            veri = json.load(f)
    except (OSError, json.JSONDecodeError) as hata:
        raise RuntimeError(f"model_config.json okunamadı ({yol}): {hata}") from hata

    for anahtar in ("model_dosyasi", "sicaklik", "guven_esigi", "saglikli_esigi", "urun_esigi"):
        if anahtar not in veri:
            raise RuntimeError(f"model_config.json'da '{anahtar}' alanı yok ({yol}).")
    return veri


yapilandirma = _yapilandirmayi_oku(YAPILANDIRMA_YOLU)

# Ortam değişkenleri testlerin (ve CI'ın) sahte bir model vermesini sağlar;
# öncelik sırası: ortam değişkeni > model_config.json > varsayılan.
CLASS_NAMES_YOLU = Path(os.getenv("CLASS_NAMES_PATH", BURASI / "class_names.json"))
MODEL_YOLU = Path(os.getenv("MODEL_PATH", BURASI / yapilandirma["model_dosyasi"]))

# Sıcaklık ölçekleme (temperature scaling): logitler softmax'tan ÖNCE bu
# sayıya bölünür. Tahmini DEĞİŞTİRMEZ (bölme sıralamayı bozmaz), sadece
# aşırı özgüvenli olasılıkları bastırıp güven skorunu gerçek isabetle
# uyumlu hale getirir.
SICAKLIK = float(yapilandirma["sicaklik"])

# Karar eşikleri. Değerler modül seviyesinde sabit olarak duruyor ki
# testler monkeypatch ile tek tek değiştirebilsin.
#
# DİKKAT: Eşik eskiden 0.95'ti, şimdi 0.70. Bu bir GEVŞETME DEĞİL --
# ölçekler farklı: 0.95 kalibre edilmemiş (aşırı özgüvenli) olasılıklar
# üzerindeydi, 0.70 ise SICAKLIK ile bastırılmış olasılıklar üzerinde.
#
# T=1.95 ve 0.70 eşiği, PlantDoc eğitim bölümünden ayrılmış 163 görüntülük
# saha VAL setinde seçildi (test setiyle phash karşılaştırılıp kopyaları
# temizlendi). 569 görüntülük test seti seçimde kullanılmadı, sadece bir kez
# raporlandı: maskeli Top-1 %69.77, kesin cevapların isabeti %88.1.
# 0.80 "sağlıklı" eşiği veriyle ayarlanmadı -- bilinçli güvenlik kararı:
# hastalıklı yaprağa "sağlıklı" demek tedaviyi geciktirir, bu hatayı
# zorlaştırmak istedik. Sınırlar ve varsayımlar: CLAUDE.md Bölüm 1.
GUVEN_ESIGI = float(yapilandirma["guven_esigi"])
SAGLIKLI_ESIGI = float(yapilandirma["saglikli_esigi"])

# Ürün tutarlılık eşiği. MASKEDEN ÖNCEKİ (tüm sınıflar açık) T ile ölçekli
# softmax'ta, seçilen ürünün sınıflarına düşen olasılık toplamı ("ürün payı")
# bu değerin altındaysa büyük ihtimalle YANLIŞ ürün seçilmiştir: model
# olasılığı başka bir ürünün sınıflarına yığmış demektir. O zaman sonuç
# engellenmez ama kullanıcıya "doğru bitkiyi mi seçtin?" diye sorulur.
#
# 0.20 eşiği 163 görüntülük saha VAL setinde seçildi (yakalama %85.6, doğru
# seçimde yanlış alarm %4.3); 569 görüntülük test seti seçimde kullanılmadı,
# bir kez raporlandı (yakalama %82.8, yanlış alarm %4.6). Pay, ürünün sınıf
# sayısından etkilenir (2 sınıflı biberin payı doğal olarak küçük kalır), bu
# yüzden tek eşik biberi dezavantajlı duruma düşürür -- ayrıntı: CLAUDE.md
# Bölüm 1 ve docs/model-yol-haritasi.md.
URUN_ESIGI = float(yapilandirma["urun_esigi"])


# --- Sınıflar ve ürün haritası ------------------------------------------
with open(CLASS_NAMES_YOLU, encoding="utf-8") as f:
    class_names = json.load(f)
num_classes = len(class_names)

# Hangi sınıf hangi ürüne ait? Eşleme sınıf ADLARININ önekinden çıkarılır.
URUN_ONEKLERI = {
    "domates": "Tomato",
    "patates": "Potato",
    "biber": "Pepper",
}


def _urun_indekslerini_cikar(sinif_adlari):
    """Her ürün için o ürüne ait sınıf indekslerini class_names'ten türetir.

    İndeksleri elle yazmak, sınıf sırası değiştiğinde SESSİZCE yanlış sonuç
    verir (maske yanlış sınıfları keser, hata hiç görünmez). Dosyadan
    türetmek bu hatayı imkânsız kılar.
    """
    indeksler = {urun: [] for urun in URUN_ONEKLERI}

    for i, ad in enumerate(sinif_adlari):
        for urun, onek in URUN_ONEKLERI.items():
            if ad.startswith(onek):
                indeksler[urun].append(i)
                break
        else:
            # Hiçbir ürüne ait olmayan sınıf: adlandırma değişmiş demektir
            # (örn. yeni model "Bell_pepper..." diyor). Sessizce geçersek
            # o sınıf hiçbir maskede görünmez ve asla tahmin edilemez.
            raise RuntimeError(
                f"'{ad}' sınıfı hiçbir ürüne eşlenemedi. Beklenen önekler: "
                f"{', '.join(URUN_ONEKLERI.values())}."
            )

    for urun, liste in indeksler.items():
        if not liste:
            raise RuntimeError(
                f"'{urun}' ürününe ait hiç sınıf bulunamadı ({CLASS_NAMES_YOLU}). "
                "Ürün seçilse bile tahmin üretilemez."
            )
    return indeksler


URUN_INDEKSLERI = _urun_indekslerini_cikar(class_names)

# main.py ve testler geçerli ürün listesini buradan okur, kendi kopyasını
# tutmaz: iki liste birbirinden ayrı düşerse hata bulmak zorlaşır.
GECERLI_URUNLER = tuple(URUN_ONEKLERI)


# --- Model --------------------------------------------------------------
# Model uygulama başlarken bir kere yüklenir, her istekte değil.
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_YOLU, map_location="cpu"))
model.eval()

# DİKKAT: Buradaki resize ve normalize değerleri eğitimdeki eval_transform ile
# birebir aynı olmalı. Farklı olursa model saçmalar.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model sınıf adlarını İngilizce döndürür (PlantVillage adları). Çiftçiye
# gösterilecek Türkçe karşılıklar burada. Listede olmayan bir ad gelirse
# (örn. yeni bir model) ham ad kullanılır, uygulama çökmez.
TURKCE_ADLAR = {
    "Pepper__bell___Bacterial_spot": "Biber - Bakteriyel leke",
    "Pepper__bell___healthy": "Biber - Sağlıklı",
    "Potato___Early_blight": "Patates - Erken yanıklık",
    "Potato___Late_blight": "Patates - Geç yanıklık (mildiyö)",
    "Potato___healthy": "Patates - Sağlıklı",
    "Tomato_Bacterial_spot": "Domates - Bakteriyel leke",
    "Tomato_Early_blight": "Domates - Erken yanıklık",
    "Tomato_Late_blight": "Domates - Geç yanıklık (mildiyö)",
    "Tomato_Leaf_Mold": "Domates - Yaprak küfü",
    "Tomato_Septoria_leaf_spot": "Domates - Septoria yaprak lekesi",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Domates - Kırmızı örümcek",
    "Tomato__Target_Spot": "Domates - Hedef leke",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Domates - Sarı yaprak kıvırcıklık virüsü",
    "Tomato__Tomato_mosaic_virus": "Domates - Mozaik virüsü",
    "Tomato_healthy": "Domates - Sağlıklı",
}


def turkce_ad(sinif):
    return TURKCE_ADLAR.get(sinif, sinif)


def _urun_maskesi(urun):
    """Seçilen ürünün DIŞINDAKİ sınıflar için True olan maske tensörü üretir.

    masked_fill bu maskenin True olduğu yerleri -inf ile doldurur; softmax
    sonrası o sınıfların olasılığı tam olarak 0 olur.
    """
    maske = torch.ones(num_classes, dtype=torch.bool)
    for i in URUN_INDEKSLERI[urun]:
        maske[i] = False
    return maske


def _adaylari_cikar(olasiliklar, urun, en_fazla=3):
    """Seçilen ürüne ait sınıflar arasından en olası en fazla 3 adayı döndürür.

    Filtre "olasılığı 0 olanları at" değil, "izinli indeksler arasından seç":
    izinli ama çok düşük olasılıklı bir sınıf float'ta 0'a yuvarlanabilir ve
    o zaman biberde aday sayısı 2 yerine 1 çıkardı.
    """
    siralanmis = sorted(URUN_INDEKSLERI[urun], key=lambda i: olasiliklar[i], reverse=True)

    return [
        {
            "hastalik": class_names[i],
            "hastalik_tr": turkce_ad(class_names[i]),
            "guven": round(olasiliklar[i] * 100, 1),
        }
        for i in siralanmis[:en_fazla]
    ]


def _esik(sinif):
    """Tahmin 'healthy' bir sınıfsa daha sıkı sağlıklı eşiğini döndürür.

    Sabitleri çağrı anında okur; böylece testler monkeypatch ile
    eşikleri değiştirebilir.
    """
    return SAGLIKLI_ESIGI if "healthy" in sinif.lower() else GUVEN_ESIGI


def tahmin_et_goruntu(img, urun):
    """Bir yaprak görüntüsü ve kullanıcının seçtiği ürün için tahmin üretir.

    Seçilen ürünün dışındaki sınıflar maskelenir, logitler sıcaklık ölçekleme
    ile kalibre edilir; güven eşiğinin altında kesin cevap yerine "emin_degil"
    ve en olası adaylar döndürülür.
    """
    if urun not in URUN_INDEKSLERI:
        raise ValueError(
            f"Geçersiz ürün: {urun!r}. Geçerli değerler: {', '.join(GECERLI_URUNLER)}."
        )

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logitler = model(x)

        # Ürün tutarlılık payı MASKEDEN ÖNCE hesaplanır: tüm sınıflar açıkken,
        # T ile ölçekli softmax'ta seçilen ürünün sınıflarına düşen olasılık
        # toplamı. Maskeden sonra bu her zaman 1.0 olurdu (maske diğerlerini
        # sıfırlar) ve hiçbir şey ölçemezdik -- bütün bilgi, modelin olasılığı
        # BAŞKA ürünlere ne kadar dağıttığında. T'ye bölme ölçümdeki gibi
        # burada da uygulanır; atlanırsa 0.20 eşiği anlamını yitirir.
        maskesiz_olasiliklar = torch.softmax(logitler / SICAKLIK, dim=1)[0]
        urun_payi = sum(maskesiz_olasiliklar[i].item() for i in URUN_INDEKSLERI[urun])
        urun_uyarisi = urun_payi < URUN_ESIGI

        # Maske softmax'tan ÖNCE uygulanmalı. Sonra uygulasaydık kesilen
        # sınıfların olasılığı pay toplamına girer, kalan olasılıklar 1'e
        # toplanmaz ve güven skorları ölçtüğümüz değerlerden sapardı.
        logitler = logitler.masked_fill(_urun_maskesi(urun), float("-inf"))

        # Sıcaklığa bölme maskeyi bozmaz: -inf / 1.95 yine -inf'tir.
        olasiliklar = torch.softmax(logitler / SICAKLIK, dim=1)

    olasiliklar = olasiliklar[0].tolist()

    # argmax'ı tüm satır üzerinde değil, izinli indeksler arasında alıyoruz:
    # maskelenenler zaten 0 ama niyeti açıkça yazmak daha güvenli.
    tahmin_idx = max(URUN_INDEKSLERI[urun], key=lambda i: olasiliklar[i])
    tahmin_sinif = class_names[tahmin_idx]
    guven = olasiliklar[tahmin_idx]
    adaylar = _adaylari_cikar(olasiliklar, urun)

    # Karşılaştırma HAM float ile; round() sadece gösterim için. Yuvarlanmış
    # değerle karşılaştırsaydık 0.6996 -> "%70, ama emin değilim" gibi
    # kendisiyle çelişen bir çıktı üretirdik.
    if guven < _esik(tahmin_sinif):
        return {
            "durum": "emin_degil",
            "urun": urun,
            "urun_uyarisi": urun_uyarisi,
            "mesaj": "Bu yaprağı net tanıyamadım. Daha yakın ve net bir fotoğraf çeker misiniz?",
            "en_yakin_tahmin": tahmin_sinif,
            "en_yakin_tahmin_tr": turkce_ad(tahmin_sinif),
            "guven": round(guven * 100, 1),
            "adaylar": adaylar,
        }
    else:
        return {
            "durum": "basarili",
            "urun": urun,
            "urun_uyarisi": urun_uyarisi,
            "hastalik": tahmin_sinif,
            "hastalik_tr": turkce_ad(tahmin_sinif),
            "guven": round(guven * 100, 1),
            "adaylar": adaylar,
        }

if __name__ == "__main__":
    import sys

    # Elle deneme: python predict.py [urun]
    urun = sys.argv[1] if len(sys.argv) > 1 else "domates"
    img = Image.open(BURASI / "test.jpg").convert("RGB")
    print(tahmin_et_goruntu(img, urun))
