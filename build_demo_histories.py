"""Build transparent sample-account histories from held-out dataset predictions."""
from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "static" / "demo_histories.json"
ACCURACY: dict[str, float] = {}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def accuracy_from(rows: list[dict[str, str]]) -> float:
    if not rows:
        raise RuntimeError("A cross-validation predictions file is empty.")
    return sum(int(row["actual_label"]) == int(row["predicted_label"]) for row in rows) / len(rows)


def choose_examples(rows: list[dict[str, str]], label: int, kind: str) -> list[dict[str, str]]:
    candidates = [r for r in rows if int(r["actual_label"]) == label and int(r["predicted_label"]) == label]
    candidates.sort(key=lambda r: float(r["parkinson_score"]))

    # Spiral data has several drawings per participant. Keep each participant once.
    unique: dict[str, dict[str, str]] = {}
    for row in candidates:
        participant = row.get("participant_id") or row.get("sample_id") or row.get("image_path")
        unique.setdefault(str(participant), row)
    candidates = list(unique.values())
    if len(candidates) < 3:
        raise RuntimeError(f"Need three correctly classified held-out examples for label {label} in {kind}; found {len(candidates)}.")
    positions = [round((len(candidates) - 1) * q) for q in (0.2, 0.5, 0.8)]
    return [candidates[i] for i in positions]


def make_item(row: dict[str, str], kind: str, when: date, dataset_label: int) -> dict:
    score = float(row["parkinson_score"])
    prediction = "Parkinson-like" if int(row["predicted_label"]) == 1 else "Healthy-like"
    if kind == "Spiral analysis":
        sample_id = Path(row["image_path"].replace("\\", "/")).name
    else:
        sample_id = row["participant_id"]
    expected = "Parkinson participant" if dataset_label == 1 else "healthy control"
    return {
        "date": f"{when.isoformat()}T12:00:00",
        "sessionId": when.isoformat(),
        "type": kind,
        "result": f"{prediction} model output · {score * 100:.1f}% PD-class score",
        "probability": score,
        "accuracy": ACCURACY[kind],
        "actualLabel": dataset_label,
        "predictedLabel": int(row["predicted_label"]),
        "sampleId": sample_id,
        "source": "research-demo",
        "sourceNote": f"Held-out public research sample · {expected} · {sample_id}",
    }


def build_account(label: int, spiral_rows: list[dict[str, str]], voice_rows: list[dict[str, str]], gait_rows: list[dict[str, str]], dates: list[date]) -> list[dict]:
    records = []
    for index, when in enumerate(dates):
        for kind, rows in (
            ("Spiral analysis", spiral_rows),
            ("Voice analysis", voice_rows),
            ("Gait analysis", gait_rows),
        ):
            selected = choose_examples(rows, label, kind)
            records.append(make_item(selected[index], kind, when, label))
    return sorted(records, key=lambda item: item["date"], reverse=True)


def main() -> None:
    today = date.today()
    dates = [today - timedelta(days=4), today - timedelta(days=2), today]
    # The deployed app loads one saved checkpoint. Use that checkpoint's own
    # held-out test predictions here, not the separate fold models' OOF file.
    spiral = read_rows(ARTIFACTS / "resnet18_checkpoint_test_predictions.csv")
    voice = read_rows(ARTIFACTS / "voice_oof_predictions.csv")
    gait = read_rows(ARTIFACTS / "gait_oof_predictions.csv")
    accuracy = {
        "Spiral analysis": float(json.loads((ARTIFACTS / "test_metrics.json").read_text(encoding="utf-8"))["accuracy"]),
        "Voice analysis": float(json.loads((ARTIFACTS / "voice_metrics.json").read_text(encoding="utf-8"))["accuracy"]),
        "Gait analysis": float(json.loads((ARTIFACTS / "gait_metrics.json").read_text(encoding="utf-8"))["accuracy"]),
    }
    global ACCURACY
    ACCURACY = accuracy
    histories = {
        "_meta": {
            "accuracy": ACCURACY,
            "selection": "Three correctly classified held-out examples per model and labeled class were selected to make the sample accounts easy to compare. These selected examples are not a representative evaluation set.",
            "dates": "Dates are illustrative display dates spaced two days apart. Each date combines separate public research participants across modalities; this is not longitudinal history for one person.",
            "score": "Accuracy-weighted average of the available model Parkinson-class scores. This is a research demonstration index, not a calibrated probability or diagnosis.",
        },
        "demo-healthy": build_account(0, spiral, voice, gait, dates),
        "demo-parkinson": build_account(1, spiral, voice, gait, dates),
    }
    OUTPUT.write_text(json.dumps(histories, indent=2), encoding="utf-8")
    print(f"Wrote {len(histories['demo-healthy'])} healthy-control and {len(histories['demo-parkinson'])} Parkinson-participant records to {OUTPUT}")
    print("Weights:", json.dumps(ACCURACY))


if __name__ == "__main__":
    main()
