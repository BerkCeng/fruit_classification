import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torchvision import datasets, models, transforms

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, 'unified_dataset')
RESNET_PATH = os.path.join(WORKSPACE_DIR, 'models', 'best_fruit_model_unified.pth')
VIT_PATH = os.path.join(WORKSPACE_DIR, 'models', 'best_fruit_model_vit.pth')
RESULTS_DIR = os.path.join(WORKSPACE_DIR, 'results')
VIT_IMG_SIZE = 128
RESNET_IMG_SIZE = 224


def evaluate_model(model, loader, device, desc):
    print(f"{desc} değerlendiriliyor...")
    y_true, y_pred = [], []
    start_t = time.time()

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(lbls.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    total_time = (time.time() - start_t) / len(loader.dataset) * 1000
    return np.array(y_true), np.array(y_pred), total_time


def save_smart_cm(y_t, y_p, class_names, title, filename):
    num_classes = len(class_names)
    cm = confusion_matrix(y_t, y_p)

    error_counts = np.zeros(num_classes)
    for i in range(len(y_t)):
        if y_t[i] != y_p[i]:
            error_counts[y_t[i]] += 1

    top_error_indices = np.argsort(error_counts)[-15:]
    top_classes = [class_names[i] for i in top_error_indices]
    sub_cm = cm[np.ix_(top_error_indices, top_error_indices)]

    plt.figure(figsize=(12, 10))
    sns.heatmap(sub_cm, annot=True, fmt='d', cmap='Blues', xticklabels=top_classes, yticklabels=top_classes)
    plt.title(title)
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def evaluate_and_compare():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Cihaz: {device}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    transform_resnet = transforms.Compose([
        transforms.Resize((RESNET_IMG_SIZE, RESNET_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_vit = transforms.Compose([
        transforms.Resize((VIT_IMG_SIZE, VIT_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset_resnet = datasets.ImageFolder(DATA_DIR, transform=transform_resnet)
    full_dataset_vit = datasets.ImageFolder(DATA_DIR, transform=transform_vit)
    class_names = full_dataset_resnet.classes
    num_classes = len(class_names)

    indices = torch.randperm(len(full_dataset_resnet)).tolist()
    test_split = int(0.1 * len(full_dataset_resnet))
    test_indices = indices[:test_split]

    test_set_rn = torch.utils.data.Subset(full_dataset_resnet, test_indices)
    test_set_vit = torch.utils.data.Subset(full_dataset_vit, test_indices)

    loader_rn = torch.utils.data.DataLoader(test_set_rn, batch_size=32, shuffle=False)
    loader_vit = torch.utils.data.DataLoader(test_set_vit, batch_size=32, shuffle=False)

    results = []

    if os.path.exists(RESNET_PATH):
        model_rn = models.resnet18()
        model_rn.fc = nn.Linear(model_rn.fc.in_features, num_classes)
        model_rn.load_state_dict(torch.load(RESNET_PATH, map_location=device))
        model_rn.to(device).eval()

        y_true, y_pred, latency = evaluate_model(model_rn, loader_rn, device, "ResNet18")
        f1 = f1_score(y_true, y_pred, average='weighted')
        acc = np.mean(y_true == y_pred)
        results.append({"Model": "ResNet18", "Accuracy": acc, "F1-Score": f1, "Latency (ms)": latency})
        save_smart_cm(y_true, y_pred, class_names, "ResNet18 Confusion Matrix (Top 15)", "resnet_confusion_matrix_top.png")

    if os.path.exists(VIT_PATH):
        model_vit = models.vit_b_16(image_size=VIT_IMG_SIZE)
        model_vit.heads.head = nn.Linear(model_vit.heads.head.in_features, num_classes)
        model_vit.load_state_dict(torch.load(VIT_PATH, map_location=device))
        model_vit.to(device).eval()

        y_true, y_pred, latency = evaluate_model(model_vit, loader_vit, device, "ViT")
        f1 = f1_score(y_true, y_pred, average='weighted')
        acc = np.mean(y_true == y_pred)
        results.append({"Model": "Vision Transformer (ViT)", "Accuracy": acc, "F1-Score": f1, "Latency (ms)": latency})
        save_smart_cm(y_true, y_pred, class_names, "ViT Confusion Matrix (Top 15)", "vit_confusion_matrix_top.png")

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(RESULTS_DIR, 'detailed_comparison.csv'), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(x="Model", y="Accuracy", data=df_results, ax=ax1, palette="viridis")
    ax1.set_title("Model Doğruluk Oranları")
    ax1.set_ylim(0.8, 1.0)
    sns.barplot(x="Model", y="Latency (ms)", data=df_results, ax=ax2, palette="magma")
    ax2.set_title("Tahmin Süresi (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparison_charts.png'))
    plt.close()

    print("Değerlendirme tamamlandı.")


if __name__ == "__main__":
    evaluate_and_compare()
