"""Test ortamının hazırlığı.

Testler gerçek modele (best_model_v3.pth) BAĞLI OLMAMALI: o dosya büyük ve
CI ortamında bulunmayabilir. Bunun yerine burada aynı mimaride (resnet18)
ama rastgele ağırlıklı SAHTE bir model üretip predict.py'ye onu gösteriyoruz.

Amaç modelin ne kadar doğru tahmin ettiğini ölçmek değil -- rastgele ağırlıkla
bu zaten ölçülemez. Amaç kodun çalıştığını doğrulamak: görüntü işleniyor mu,
dönen sözlüğün alanları doğru mu, güven eşiği beklendiği gibi davranıyor mu.

Bu dosya pytest tarafından test modüllerinden ÖNCE çalıştırılır; ortam
değişkenlerini burada kurmamızın sebebi de bu (predict.py modeli import
anında yüklüyor).
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

KOK = Path(__file__).resolve().parents[1]
BACKEND = KOK / "bitki-backend"

# predict.py ve main.py'yi import edebilmek için backend klasörünü yola ekle
sys.path.insert(0, str(BACKEND))

# Gerçek modeldeki gibi birkaç sınıf; isimler gerçek class_names.json ile
# aynı olmak zorunda değil, sadece biçim olarak benzer olmalı.
SAHTE_SINIFLAR = [
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Tomato_Leaf_Mold",
]

_GECICI_KLASOR = Path(tempfile.mkdtemp(prefix="sahte-model-"))


def _sahte_ortam_kur():
    sinif_yolu = _GECICI_KLASOR / "class_names.json"
    sinif_yolu.write_text(json.dumps(SAHTE_SINIFLAR), encoding="utf-8")

    # predict.py'nin kurduğu mimarinin AYNISI, yoksa load_state_dict patlar.
    sahte_model = models.resnet18(weights=None)
    sahte_model.fc = nn.Linear(sahte_model.fc.in_features, len(SAHTE_SINIFLAR))

    model_yolu = _GECICI_KLASOR / "sahte_model.pth"
    torch.save(sahte_model.state_dict(), model_yolu)

    os.environ["CLASS_NAMES_PATH"] = str(sinif_yolu)
    os.environ["MODEL_PATH"] = str(model_yolu)


_sahte_ortam_kur()


def pytest_sessionfinish(session, exitstatus):
    """Testler bitince geçici sahte model klasörünü sil."""
    shutil.rmtree(_GECICI_KLASOR, ignore_errors=True)
