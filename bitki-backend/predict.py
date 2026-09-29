import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Yollar bu dosyanın bulunduğu klasöre göre çözülür; böylece uygulama
# hangi klasörden başlatılırsa başlatılsın model dosyaları bulunur.
# Ortam değişkenleri testlerin (ve CI'ın) sahte bir model vermesini sağlar.
BURASI = Path(__file__).resolve().parent
CLASS_NAMES_YOLU = Path(os.getenv("CLASS_NAMES_PATH", BURASI / "class_names.json"))
MODEL_YOLU = Path(os.getenv("MODEL_PATH", BURASI / "best_model_v3.pth"))

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
GUVEN_ESIGI = 0.95

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
            "guven": round(guven * 100, 1)
        }
    else:
        return {
            "durum": "basarili",
            "hastalik": tahmin_sinif,
            "guven": round(guven * 100, 1)
        }

if __name__ == "__main__":
    img = Image.open(BURASI / "test.jpg").convert("RGB")
    print(tahmin_et_goruntu(img))
