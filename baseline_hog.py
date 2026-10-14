
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

CSV_PATH = ROOT / "NewSpiral.csv"

# IMPORTANT:
# Confirm these values against the dataset's official metadata first.
# Do not guess or swap them merely to improve the results.
CLASS_TYPE_PARKINSON = 2
CLASS_TYPE_HEALTHY = 1

if CLASS_TYPE_PARKINSON == CLASS_TYPE_HEALTHY:
    raise ValueError("Healthy and Parkinson class values must differ.")

df = pd.read_csv(CSV_PATH, dtype={"ID_PATIENT": str})
df["CLASS_TYPE"] = pd.to_numeric(df["CLASS_TYPE"], errors="raise").astype(int)

observed = set(df["CLASS_TYPE"].unique())
expected = {CLASS_TYPE_HEALTHY, CLASS_TYPE_PARKINSON}
if observed != expected:
    raise ValueError(
        f"Expected CLASS_TYPE values {expected}, found {observed}. "
        "Check the dataset metadata and mapping."
    )

# Label 0 = healthy; label 1 = Parkinson, according to the mapping above.
df["label"] = (df["CLASS_TYPE"] == CLASS_TYPE_PARKINSON).astype(int)

def locate(name, class_type):
    folder = (
        "PatientSpiral"
        if class_type == CLASS_TYPE_PARKINSON
        else "HealthySpiral"
    )
    base = ROOT / folder

    candidates = [str(name)]
    if class_type == CLASS_TYPE_PARKINSON and str(name).startswith("mea"):
        candidates.append("sp" + str(name)[3:])

    for candidate in candidates:
        matches = list(base.rglob(candidate))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Cannot find {name!r} for CLASS_TYPE={class_type} in {base}"
    )

df["path"] = [
    locate(row.IMAGE_NAME, int(row.CLASS_TYPE))
    for _, row in df.iterrows()
]

print("Images found:", len(df))
print("Label counts (0=Healthy, 1=Parkinson):")
print(df["label"].value_counts().sort_index())
print("Participants:", df["ID_PATIENT"].nunique())

def extract_features(path):
    image = (
        np.asarray(
            Image.open(path).convert("L").resize((256, 256)),
            dtype=np.float32
        ) / 255.0
    )
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

X = np.vstack([extract_features(path) for path in df["path"]])
y = df["label"].to_numpy()
groups = df["ID_PATIENT"].to_numpy()

# One fixed participant-level holdout. No participant appears in both sets.
splitter = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

train_people = set(groups[train_idx])
test_people = set(groups[test_idx])
print("Participants in both sets:", train_people & test_people)
print("Train participants:", len(train_people))
print("Test participants:", len(test_people))
assert train_people.isdisjoint(test_people), "Participant leakage detected!"

model = Pipeline([
    ("scale", StandardScaler()),
    ("svc", SVC(
        kernel="rbf",
        C=2,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X[train_idx], y[train_idx])

pred = model.predict(X[test_idx])
probabilities = model.predict_proba(X[test_idx])
positive_col = list(model.classes_).index(1)
prob = probabilities[:, positive_col]

print("\nHOG + SVM participant-held-out test results")
print("Accuracy:", round(accuracy_score(y[test_idx], pred), 3))
print("Precision:", round(precision_score(y[test_idx], pred, zero_division=0), 3))
print("Recall:", round(recall_score(y[test_idx], pred, zero_division=0), 3))
print("F1:", round(f1_score(y[test_idx], pred, zero_division=0), 3))
print("ROC-AUC:", round(roc_auc_score(y[test_idx], prob), 3))
print("Confusion matrix (rows=actual, columns=predicted; 0=Healthy, 1=Parkinson):")
print(confusion_matrix(y[test_idx], pred, labels=[0, 1]))
print(classification_report(
    y[test_idx], pred,
    labels=[0, 1],
    target_names=["Healthy", "Parkinson"],
    zero_division=0
))

# Save only after the label mapping is verified and evaluation is reviewed.
# Refit on all data for the final prototype model.
model.fit(X, y)
joblib.dump(model, ART / "hog_svm.joblib")
print("Saved:", ART / "hog_svm.joblib")
