import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, 'unified_dataset')
MODEL_SAVE_PATH = os.path.join(WORKSPACE_DIR, 'models', 'best_fruit_model_vit.pth')
CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, 'models', 'checkpoint_vit.pth')

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
IMG_SIZE = 128


def resize_pos_embed(model, new_size):
    pos_embed = model.encoder.pos_embedding
    cls_token = pos_embed[:, :1, :]
    grid_embed = pos_embed[:, 1:, :]

    old_grid_size = int(grid_embed.shape[1] ** 0.5)
    new_grid_size = new_size // 16

    grid_embed = grid_embed.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
    grid_embed = F.interpolate(grid_embed, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
    grid_embed = grid_embed.permute(0, 2, 3, 1).reshape(1, -1, pos_embed.shape[-1])

    new_pos_embed = torch.cat((cls_token, grid_embed), dim=1)
    model.encoder.pos_embedding = nn.Parameter(new_pos_embed)
    model.image_size = new_size
    model.patch_size = 16


def train_vit():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Cihaz: {device}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATA_DIR):
        print(f"Veri dizini bulunamadı: {DATA_DIR}")
        return

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    num_classes = len(full_dataset.classes)
    print(f"Sınıf sayısı: {num_classes} | Eğitim: {train_size} | Doğrulama: {val_size}")

    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    resize_pos_embed(model, IMG_SIZE)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    start_epoch = 0
    best_acc = 0.0

    if os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            print(f"Kontrol noktası yüklendi (Epoch {start_epoch}).")
        except Exception as e:
            print(f"Kontrol noktası okunamadı: {e}")

    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for i, (images, labels) in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            if (i + 1) % 500 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc
                }, CHECKPOINT_PATH)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        scheduler.step()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': max(acc, best_acc)
        }, CHECKPOINT_PATH)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f} | Acc: %{acc:.2f} | Süre: {time.time()-start_time:.1f}s")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model kaydedildi: %{best_acc:.2f}")

    print(f"Eğitim tamamlandı. En iyi doğruluk: %{best_acc:.2f}")


if __name__ == "__main__":
    train_vit()
