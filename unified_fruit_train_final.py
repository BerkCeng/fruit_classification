import logging
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, 'unified_dataset')
MODEL_PATH = os.path.join(WORKSPACE_DIR, 'models', 'best_fruit_model_unified.pth')
RESULTS_DIR = os.path.join(WORKSPACE_DIR, 'results')

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-4

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


def save_visual_results(history, model, test_loader, class_names):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss', color='red', marker='o')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='Test Accuracy', color='blue', marker='o')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'egitim_grafikleri.png'))
    plt.close()

    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:6].to(device), labels[:6].to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(15, 10))
    for i in range(6):
        plt.subplot(2, 3, i + 1)

        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        pred_label = class_names[preds[i]]
        true_label = class_names[labels[i]]
        color = 'green' if preds[i] == labels[i] else 'red'

        plt.imshow(img)
        plt.title(f"Tahmin: {pred_label}\nGerçek: {true_label}", color=color, fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'tahmin_ornekleri.png'))
    plt.close()


def train_unified():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(100),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATA_DIR):
        logging.error(f"Veri dizini bulunamadı: {DATA_DIR}")
        return

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    dataset = datasets.ImageFolder(DATA_DIR, transform)
    num_classes = len(dataset.classes)
    class_names = dataset.classes

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    logging.info(f"Sınıf sayısı: {num_classes} | Eğitim: {train_size} | Test: {test_size}")

    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            logging.info("Mevcut model ağırlıkları yüklendi.")
        except Exception:
            logging.warning("Mevcut ağırlıklar uyumsuz, ImageNet ağırlıklarıyla devam ediliyor.")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'loss': [], 'acc': []}
    best_acc = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            if (i + 1) % 50 == 0:
                logging.info(f"Epoch {epoch+1}/{EPOCHS} | Batch {i+1} | Loss: {loss.item():.4f}")

        model.eval()
        test_corrects = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_corrects += torch.sum(preds == labels.data)

        epoch_acc = (test_corrects.float() / test_size) * 100
        epoch_loss = running_loss / train_size

        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc.item())

        duration = time.time() - start_time
        logging.info(f"Epoch {epoch+1} | Test Acc: %{epoch_acc:.2f} | Loss: {epoch_loss:.4f} | Süre: {duration:.1f}s")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), MODEL_PATH)
            logging.info(f"Model kaydedildi: %{best_acc:.2f}")

    save_visual_results(history, model, test_loader, class_names)
    logging.info(f"Eğitim tamamlandı. En iyi doğruluk: %{best_acc:.2f}")


if __name__ == '__main__':
    train_unified()
