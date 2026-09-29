import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

with open("class_names.json") as f:
    class_names = json.load(f)
num_classes = len(class_names)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("best_model_v3.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

GUVEN_ESIGI = 0.60

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
    img = Image.open("test.jpg").convert("RGB")
    print(tahmin_et_goruntu(img))