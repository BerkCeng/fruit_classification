import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORKSPACE_DIR, 'models', 'best_fruit_model_unified.pth')
DATA_DIR = os.path.join(WORKSPACE_DIR, 'unified_dataset')
TEST_DIR = os.path.join(WORKSPACE_DIR, 'test_image')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'tahmin_sonuclari')


def run_batch_prediction():
    if not os.path.exists(MODEL_PATH):
        print(f"Model dosyası bulunamadı: {MODEL_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])[:268]

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Cihaz: {device} | Sınıf sayısı: {len(class_names)}")

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"{len(images)} görsel işleniyor...")

    for img_name in sorted(images):
        img_path = os.path.join(TEST_DIR, img_name)
        try:
            original_img = Image.open(img_path).convert('RGB')
            input_tensor = transform(original_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top3_prob, top3_indices = torch.topk(probabilities, 3)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [1.5, 1]})

            ax1.imshow(original_img)
            ax1.axis('off')

            best_name = class_names[top3_indices[0][0]]
            best_conf = top3_prob[0][0].item() * 100
            label_text = f"{best_name} %{best_conf:.1f}"

            ax1.text(
                10, 10, label_text, color='white', fontsize=15, fontweight='bold',
                bbox=dict(facecolor='green' if best_conf > 70 else 'orange', alpha=0.8, edgecolor='none', pad=10)
            )
            ax1.set_title(img_name, fontsize=12, pad=20)

            y_pos = np.arange(3)
            confs = [p.item() * 100 for p in top3_prob[0]]
            names = [class_names[idx] for idx in top3_indices[0]]

            bars = ax2.barh(y_pos, confs, align='center', color=['#2ecc71', '#3498db', '#95a5a6'])
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(names, fontsize=12, fontweight='bold')
            ax2.invert_yaxis()
            ax2.set_xlabel('Güven Oranı (%)', fontsize=12)
            ax2.set_title('Top 3 Tahmin', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 100)

            for bar in bars:
                width = bar.get_width()
                ax2.text(
                    width + 1, bar.get_y() + bar.get_height() / 2, f"%{width:.1f}",
                    va='center', fontsize=10, fontweight='bold'
                )

            plt.tight_layout()
            save_path = os.path.join(OUTPUT_DIR, f"result_{img_name}")
            plt.savefig(save_path)
            plt.close()

            print(f"{img_name} -> {best_name} (%{best_conf:.1f})")

        except Exception as e:
            print(f"Hata ({img_name}): {e}")


if __name__ == '__main__':
    run_batch_prediction()
