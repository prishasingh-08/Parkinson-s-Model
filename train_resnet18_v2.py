
from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "NewSpiral.csv"
OUTPUT_DIR = ROOT / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
FOLDS = 5
IMAGE_SIZE = 224
BATCH_SIZE = 16
MAX_EPOCHS = 20
PATIENCE = 4
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

# ---------------- LOAD DATA ----------------

df = pd.read_csv(CSV_PATH)
records = []

for _, row in df.iterrows():
    class_type = int(row["CLASS_TYPE"])

    if class_type not in (1, 2):
        continue

    folder = "PatientSpiral" if class_type == 2 else "HealthySpiral"
    name = str(row["IMAGE_NAME"])
    candidates = [name]

    # Some CSV entries use mea... names while files use sp...
    if folder == "PatientSpiral" and name.lower().startswith("mea"):
        candidates.append("sp" + name[3:])

    image_path = next(
        (ROOT / folder / candidate for candidate in candidates
         if (ROOT / folder / candidate).is_file()),
        None,
    )

    if image_path is None:
        print("Missing image:", name, "in", folder)
        continue

    records.append({
        "path": str(image_path),
        "label": 1 if class_type == 2 else 0,
        "group": row["ID_PATIENT"],
    })

data = pd.DataFrame(records)

print("Images:", len(data))
print("Participants:", data["group"].nunique())
print("Labels (0=Healthy, 1=Parkinson):")
print(data["label"].value_counts().sort_index())

# ---------------- TRANSFORMS ----------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(8),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.04, 0.04),
        scale=(0.95, 1.05),
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class SpiralDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label


def make_model():
    # Uses ImageNet-pretrained weights. The first run may need internet
    # access to download them.
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Could not load pretrained ResNet18 weights. "
            "Check your internet connection and try again."
        ) from exc

    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


def evaluate(model, loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return (
        np.asarray(all_labels),
        np.asarray(all_preds),
        np.asarray(all_probs),
    )


def train_fold(train_frame, val_frame, test_frame, fold):
    train_loader = DataLoader(
        SpiralDataset(train_frame, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        SpiralDataset(val_frame, eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        SpiralDataset(test_frame, eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = make_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_count = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_total += loss.item() * len(labels)

        val_loss = val_loss_total / len(val_frame)
        train_loss = running_loss / len(train_frame)

        print(
            f"Fold {fold} | Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)

    labels, preds, probs = evaluate(model, test_loader)

    return labels, preds, probs, best_epoch


def main():
    # Each outer test fold contains unseen participants.
    outer = StratifiedGroupKFold(
        n_splits=FOLDS, shuffle=True, random_state=SEED
    )

    oof = []
    best_epochs = []

    for fold, (development_idx, test_idx) in enumerate(
        outer.split(data, data["label"], groups=data["group"]),
        start=1,
    ):
        development = data.iloc[development_idx].reset_index(drop=True)
        test_frame = data.iloc[test_idx].reset_index(drop=True)

        # Separate participants for early stopping, inside development data.
        inner = StratifiedGroupKFold(
            n_splits=4, shuffle=True, random_state=SEED + fold
        )

        fit_idx, val_idx = next(
            inner.split(
                development,
                development["label"],
                groups=development["group"],
            )
        )

        train_frame = development.iloc[fit_idx].reset_index(drop=True)
        val_frame = development.iloc[val_idx].reset_index(drop=True)

        assert set(train_frame["group"]).isdisjoint(val_frame["group"])
        assert set(development["group"]).isdisjoint(test_frame["group"])

        print(f"\n========== OUTER FOLD {fold}/{FOLDS} ==========")
        print(
            "Participants: train =", train_frame["group"].nunique(),
            "| validation =", val_frame["group"].nunique(),
            "| test =", test_frame["group"].nunique(),
        )

        labels, preds, probs, best_epoch = train_fold(
            train_frame, val_frame, test_frame, fold
        )
        best_epochs.append(best_epoch)

        for path, group, y, pred, prob in zip(
            test_frame["path"],
            test_frame["group"],
            labels,
            preds,
            probs,
        ):
            oof.append({
                "image_path": path,
                "participant_id": group,
                "actual_label": int(y),
                "predicted_label": int(pred),
                "parkinson_score": float(prob),
                "fold": fold,
            })

        print("Fold accuracy:", round(accuracy_score(labels, preds), 3))

    result = pd.DataFrame(oof)
    result.to_csv(OUTPUT_DIR / "resnet18_oof_predictions.csv", index=False)

    y = result["actual_label"].to_numpy()
    pred = result["predicted_label"].to_numpy()
    prob = result["parkinson_score"].to_numpy()

    print("\n========== PARTICIPANT-GROUPED OOF RESULTS ==========")
    print("Images evaluated:", len(result))
    print("Participants evaluated:", result["participant_id"].nunique())
    print(f"Accuracy:  {accuracy_score(y, pred):.3f}")
    print(f"Precision: {precision_score(y, pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y, pred, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y, pred, zero_division=0):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y, prob):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))
    print("Best epochs by fold:", best_epochs)
    print("Out-of-fold predictions saved to:")
    print(OUTPUT_DIR / "resnet18_oof_predictions.csv")
    print("\nThis is research evaluation, not clinical validation.")


if __name__ == "__main__":
    main()
