import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

with open("class_names.json") as f:
    class_names = json.load(f)
num_classes = len(class_names)
print(f"{num_classes} sınıf yüklendi.")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)  


model.load_state_dict(torch.load("best_model_v3.pth", map_location="cpu"))
model.eval()
print("Model yüklendi.")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open("test.jpg").convert("RGB")
x = transform(img).unsqueeze(0)
with torch.no_grad():   
    output = model(x)              
    probs = torch.softmax(output, dim=1)
    guven, tahmin_idx = torch.max(probs, 1) 

tahmin_sinif = class_names[tahmin_idx.item()]
print(f"\nTahmin: {tahmin_sinif}")
print(f"Güven: {guven.item()*100:.1f}%")