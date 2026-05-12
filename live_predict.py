import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, 'unified_dataset')
MODEL_PATH = os.path.join(WORKSPACE_DIR, 'models', 'best_fruit_model_unified.pth')


def predict_fruit(image_path):
    if not os.path.exists(DATA_DIR):
        print(f"Veri dizini bulunamadı: {DATA_DIR}")
        return

    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])[:268]
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Görsel okunamadı: {e}")
        return

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)

        result = class_names[preds[0]]
        conf_score = confidence[0].item() * 100

        print(f"Tahmin: {result} (%{conf_score:.2f})")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Kullanım: python3 live_predict.py <gorsel_yolu>")
    else:
        predict_fruit(sys.argv[1])
