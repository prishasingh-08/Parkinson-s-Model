
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "NewSpiral.csv")


def find_image(row):
    folder = "PatientSpiral" if int(row["CLASS_TYPE"]) == 2 else "HealthySpiral"
    name = str(row["IMAGE_NAME"])
    candidates = [name]

    if folder == "PatientSpiral" and name.lower().startswith("mea"):
        candidates.append("sp" + name[3:])

    for candidate in candidates:
        path = ROOT / folder / candidate
        if path.is_file():
            return path
    return None


X, y, groups = [], [], []

for _, row in df.iterrows():
    if int(row["CLASS_TYPE"]) not in (1, 2):
        continue

    path = find_image(row)
    if path is None:
        continue

    image = np.array(
        Image.open(path).convert("L").resize((256, 256))
    ) / 255.0

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )

    X.append(features)
    y.append(1 if int(row["CLASS_TYPE"]) == 2 else 0)
    groups.append(row["ID_PATIENT"])

X = np.asarray(X)
y = np.asarray(y)
groups = np.asarray(groups)

model_template = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(
        kernel="rbf",
        C=2,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )),
])

splitter = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)

all_y, all_pred, all_prob = [], [], []

for fold, (train_idx, test_idx) in enumerate(
    splitter.split(X, y, groups=groups), start=1
):
    model = clone(model_template)
    model.fit(X[train_idx], y[train_idx])

    pred = model.predict(X[test_idx])
    prob = model.predict_proba(X[test_idx])[:, list(model.classes_).index(1)]

    all_y.extend(y[test_idx])
    all_pred.extend(pred)
    all_prob.extend(prob)

    print(f"Fold {fold}: accuracy = {accuracy_score(y[test_idx], pred):.3f}")

all_y = np.asarray(all_y)
all_pred = np.asarray(all_pred)
all_prob = np.asarray(all_prob)

print("\n=== PARTICIPANT-GROUPED CROSS-VALIDATION ===")
print("Images evaluated:", len(all_y))
print("Participants:", len(np.unique(groups)))
print(f"Accuracy:  {accuracy_score(all_y, all_pred):.3f}")
print(f"Precision: {precision_score(all_y, all_pred, zero_division=0):.3f}")
print(f"Recall:    {recall_score(all_y, all_pred, zero_division=0):.3f}")
print(f"F1-score:  {f1_score(all_y, all_pred, zero_division=0):.3f}")
print(f"ROC-AUC:   {roc_auc_score(all_y, all_prob):.3f}")
print("Confusion matrix:")
print(confusion_matrix(all_y, all_pred))
