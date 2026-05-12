import os
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, 'unified_dataset')
MODEL_PATH = os.path.join(WORKSPACE_DIR, 'models', 'best_fruit_model_unified.pth')
RESULTS_DIR = os.path.join(WORKSPACE_DIR, 'results')
BATCH_SIZE = 64
IMG_SIZE = 224


def evaluate_resnet():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Cihaz: {device}")

    if not os.path.exists(MODEL_PATH):
        print(f"Model bulunamadı: {MODEL_PATH}")
        return

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATA_DIR):
        print(f"Veri seti bulunamadı: {DATA_DIR}")
        return

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    class_names = full_dataset.classes

    _, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [int(0.9 * len(full_dataset)), len(full_dataset) - int(0.9 * len(full_dataset))]
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()

    y_true, y_pred = [], []
    start_t = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    total_time = (time.time() - start_t) * 1000
    avg_latency = total_time / len(test_dataset)
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: %{accuracy*100:.2f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Ortalama Gecikme: {avg_latency:.2f} ms")

    cm = confusion_matrix(y_true, y_pred)
    error_counts = np.zeros(len(class_names))
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            error_counts[y_true[i]] += 1

    top_error_indices = np.argsort(error_counts)[-15:]
    top_classes = [class_names[i] for i in top_error_indices]
    sub_cm = cm[np.ix_(top_error_indices, top_error_indices)]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(sub_cm, annot=True, fmt='d', cmap='Reds', xticklabels=top_classes, yticklabels=top_classes)
    plt.title("ResNet18 Confusion Matrix (Top 15)")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.tight_layout()

    cm_path = os.path.join(RESULTS_DIR, 'resnet_confusion_matrix_top.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Kaydedildi: {cm_path}")


if __name__ == "__main__":
    evaluate_resnet()
