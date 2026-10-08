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
# Hangi ağırlık dosyasının yükleneceği KODUN değil MODELİN bilgisi. Dosya adını
# kodun içine gömmek, model değiştiğinde kod değiştirmek demekti; ayrı bir
# "model künyesi" dosyasında tutunca model ve kod birbirinden ayrışıyor
# (class_names.json gibi).
YAPILANDIRMA_YOLU = BURASI / "model_config.json"


def _yapilandirmayi_oku(yol):
    """model_config.json'u okur ve beklenen alanların varlığını doğrular.

    Dosya bozuk veya eksikse açılışta hata verir: yarım yapılandırmayla
    tahmin üretmek, hiç açılmamaktan daha kötü.
    """
    try:
        with open(yol, encoding="utf-8") as f:
            veri = json.load(f)
    except (OSError, json.JSONDecodeError) as hata:
        raise RuntimeError(f"model_config.json okunamadı ({yol}): {hata}") from hata

    for anahtar in ("model_dosyasi",):
        if anahtar not in veri:
            raise RuntimeError(f"model_config.json'da '{anahtar}' alanı yok ({yol}).")
    return veri


yapilandirma = _yapilandirmayi_oku(YAPILANDIRMA_YOLU)

# Ortam değişkenleri testlerin (ve CI'ın) sahte bir model vermesini sağlar;
# öncelik sırası: ortam değişkeni > model_config.json > varsayılan.
CLASS_NAMES_YOLU = Path(os.getenv("CLASS_NAMES_PATH", BURASI / "class_names.json"))
MODEL_YOLU = Path(os.getenv("MODEL_PATH", BURASI / yapilandirma["model_dosyasi"]))

with open(CLASS_NAMES_YOLU, encoding="utf-8") as f:
    class_names = json.load(f)
num_classes = len(class_names)

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

# Güven eşiği veriyle seçildi (2026-09-29, tune_threshold.py ile ölçüldü).
# PlantDoc test setinde 569 gerçek tarla görüntüsü üzerinde ölçüm:
#   eşik 0.80 -> kapsama %58.3, isabet %73.8
#   eşik 0.90 -> kapsama %45.0, isabet %80.1
#   eşik 0.95 -> kapsama %36.6, isabet %84.6   <-- seçilen
#   eşik 0.97 -> kapsama %30.2, isabet %87.8
# Kural: isabetin ~%85'e ulaştığı EN DÜŞÜK eşik. 0.97 daha isabetli ama
# 6.4 puan daha kapsama yakıyor; kazanç bu bedeli karşılamıyor.
# Bedeli bilerek kabul ediyoruz: fotoğrafların ~%63'üne "emin değil" denir.
# Modelin eşiksiz genel doğruluğu bu sette %62.2 -- yanlış yönlendirmektense
# susmak tercih edildi.
# UYARI: Eşik, yukarıdaki rakamların ölçüldüğü AYNI test setine bakılarak
# seçildi. Bu yüzden %84.6 isabet iyimser bir tahmin; görülmemiş yeni
# fotoğraflarda biraz daha düşük çıkması beklenir. Dürüst bir ölçüm için
# eşik ayrı bir doğrulama setinde seçilip test setinde bir kez ölçülmeli.
GUVEN_ESIGI = 0.95

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


def tahmin_et_goruntu(img):
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        guven, tahmin_idx = torch.max(probs, 1)

    guven = guven.item()
    tahmin_sinif = class_names[tahmin_idx.item()]

    if guven < GUVEN_ESIGI:
        return {
            "durum": "emin_degil",
            "mesaj": "Bu yaprağı net tanıyamadım. Daha yakın ve net bir fotoğraf çeker misiniz?",
            "en_yakin_tahmin": tahmin_sinif,
            "en_yakin_tahmin_tr": turkce_ad(tahmin_sinif),
            "guven": round(guven * 100, 1)
        }
    else:
        return {
            "durum": "basarili",
            "hastalik": tahmin_sinif,
            "hastalik_tr": turkce_ad(tahmin_sinif),
            "guven": round(guven * 100, 1)
        }

if __name__ == "__main__":
    img = Image.open(BURASI / "test.jpg").convert("RGB")
    print(tahmin_et_goruntu(img))
