import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Config ────────────────────────────────────────────────
DATASET_DIR = 'dataset_pinn'
BATCH_SIZE  = 16
EPOCHS      = 20
LR          = 0.001
IMG_SIZE    = 224
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Transforms ────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Dataset split ─────────────────────────────────────────
full_dataset = datasets.ImageFolder(DATASET_DIR)
classes      = full_dataset.classes
print(f"Classes: {classes}")

total   = len(full_dataset)
t_size  = int(0.7 * total)
v_size  = int(0.15 * total)
te_size = total - t_size - v_size

train_ds, val_ds, test_ds = torch.utils.data.random_split(
    full_dataset, [t_size, v_size, te_size],
    generator=torch.Generator().manual_seed(42)
)

train_ds.dataset.transform = train_transforms
val_ds.dataset.transform   = val_transforms
test_ds.dataset.transform  = val_transforms

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# ── Model: ResNet18 + Transfer Learning ───────────────────
model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze all layers except final
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for 3 classes
model.fc = nn.Linear(model.fc.in_features, 3)

model = model.to(DEVICE)

# ── Loss & Optimizer ──────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ── Training loop ─────────────────────────────────────────
train_losses, val_losses     = [], []
train_accs,   val_accs       = [], []

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    t_loss, t_correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        t_loss    += loss.item() * imgs.size(0)
        t_correct += (outputs.argmax(1) == labels).sum().item()

    # --- Validate ---
    model.eval()
    v_loss, v_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs  = model(imgs)
            loss     = criterion(outputs, labels)
            v_loss    += loss.item() * imgs.size(0)
            v_correct += (outputs.argmax(1) == labels).sum().item()

    scheduler.step()

    t_loss /= len(train_ds)
    v_loss /= len(val_ds)
    t_acc   = t_correct / len(train_ds)
    v_acc   = v_correct / len(val_ds)

    train_losses.append(t_loss)
    val_losses.append(v_loss)
    train_accs.append(t_acc)
    val_accs.append(v_acc)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
          f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")

# ── Save model ────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/airfoil_pinn_cnn.pth')
print("\n✅ Model saved to models/airfoil_cnn.pth")

# ── Plot training curves ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses,   label='Val Loss')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid(True)

ax2.plot(train_accs, label='Train Acc')
ax2.plot(val_accs,   label='Val Acc')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('models/training_curves.png', dpi=150)
print("✅ Training curves saved to models/training_curves.png")

# ── Test evaluation ───────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs   = imgs.to(DEVICE)
        preds  = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n── Classification Report ──────────────────────────")
print(classification_report(all_labels, all_preds, target_names=classes))

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(classes, rotation=45)
ax.set_yticklabels(classes)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=12)
plt.colorbar(im)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150)
print("✅ Confusion matrix saved to models/confusion_matrix.png")