"""Train/evaluate a research baseline from public PhysioNet GaitPDB files."""
from __future__ import annotations

import json
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from gait_model import extract_gait_features

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "research_data" / "gaitpdb"
ARTIFACTS = ROOT / "artifacts"


def main():
    rows, labels, groups = [], [], []
    skipped = []
    for path in sorted(DATA.glob("*_01.txt")):
        name = path.stem
        if "Pt" not in name and "Co" not in name:
            continue
        try:
            raw = np.loadtxt(path, dtype=np.float64)
        except Exception:
            skipped.append(name)
            continue
        if raw.ndim != 2 or raw.shape[1] != 19 or raw.shape[0] < 500 or not np.isfinite(raw).all() or not np.all(np.diff(raw[:, 0]) > 0):
            skipped.append(name)
            continue
        rows.append(extract_gait_features(raw[:, 1:]))
        labels.append(int("Pt" in name))
        groups.append(name)
    X, y = np.vstack(rows), np.asarray(labels, dtype=np.int64)
    counts = np.bincount(y, minlength=2)
    if len(y) < 20 or len(np.unique(y)) != 2 or min(counts) < 5:
        raise RuntimeError(f"Need at least 20 valid labeled participants and 5+ examples per class; found {len(y)} (healthy={counts[0]}, Parkinson={counts[1]}). Skipped: {', '.join(skipped)}")
    candidates = tuple(k for k in (6, 10, 16, 24, 32, 48) if k <= X.shape[1])

    def make_model(k, seed):
        return make_pipeline(
            StandardScaler(),
            SelectKBest(f_classif, k=k),
            SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=seed),
        )

    # Keep every outer test fold out of feature selection and model fitting.
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261019)
    probs = np.zeros(len(y), dtype=np.float64)
    selected_by_fold = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261020 + fold)
        scores = {}
        for k in candidates:
            inner_probs = cross_val_predict(make_model(k, 20261020 + fold), X[train_idx], y[train_idx], cv=inner, method="predict_proba")[:, 1]
            scores[k] = float(roc_auc_score(y[train_idx], inner_probs))
        fold_k = max(scores, key=scores.get)
        selected_by_fold.append(int(fold_k))
        fold_model = make_model(fold_k, 20261020 + fold).fit(X[train_idx], y[train_idx])
        probs[test_idx] = fold_model.predict_proba(X[test_idx])[:, list(fold_model.classes_).index(1)]
    pred = (probs >= .5).astype(int)

    # Pick a final deployment feature count using all training participants.
    final_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261019)
    final_scores = {}
    for k in candidates:
        final_probs = cross_val_predict(make_model(k, 20261019), X, y, cv=final_cv, method="predict_proba")[:, 1]
        final_scores[k] = float(roc_auc_score(y, final_probs))
    best_k = max(final_scores, key=final_scores.get)
    model = make_model(best_k, 20261019)
    metrics = {
        "dataset": "PhysioNet Gait in Parkinson's Disease (GaitPDB), Open Data Commons Attribution 1.0",
        "source": "https://physionet.org/content/gaitpdb/1.0.0/",
        "input": "100 Hz insole vertical-force recording, 19 columns (time + 18 force channels)",
        "subjects": int(len(y)), "healthy": int((y == 0).sum()), "parkinson": int((y == 1).sum()),
        "validation": "Nested 5-fold participant-level out-of-fold evaluation; feature count selected only within each outer training fold; one usual-walk record per participant",
        "selected_features": int(best_k), "outer_fold_selected_features": selected_by_fold,
        "final_fit_feature_count_selection_cv_roc_auc": final_scores,
        "accuracy": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "confusion_matrix_rows_healthy_parkinson_cols_healthy_parkinson": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "note": "Small-sample research baseline only. Nested cross-validation is an estimate, not external clinical validation. It requires compatible insole force sensors and is not a diagnosis; results may not generalize to other sensors or populations.",
    }
    model.fit(X, y)
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"pipeline": model, "class_names": ["Healthy", "Parkinson"], "feature_count": X.shape[1]}, ARTIFACTS / "gait_model.joblib")
    (ARTIFACTS / "gait_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (ARTIFACTS / "gait_oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["participant_id", "actual_label", "predicted_label", "parkinson_score"])
        writer.writerows(zip(groups, y.tolist(), pred.tolist(), probs.tolist()))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
