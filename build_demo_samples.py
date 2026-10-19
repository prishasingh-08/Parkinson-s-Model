"""Materialize one real input per class and modality for classroom upload demos."""
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "static" / "demo_samples"
DEST.mkdir(parents=True, exist_ok=True)
histories = json.loads((ROOT / "static" / "demo_histories.json").read_text(encoding="utf-8"))

for account, label, class_name in (("demo-healthy", 0, "healthy"), ("demo-parkinson", 1, "parkinson")):
    latest = {item["type"]: item for item in histories[account] if item["sessionId"] == max(x["sessionId"] for x in histories[account])}
    spiral = latest["Spiral analysis"]["sampleId"]
    spiral_root = "HealthySpiral" if label == 0 else "PatientSpiral"
    shutil.copy2(ROOT / spiral_root / spiral, DEST / f"spiral_{class_name}.jpg")

    voice_id = latest["Voice analysis"]["sampleId"]
    voice_archive = ROOT / "research_data" / ("voice_HC_AH.zip" if label == 0 else "voice_PD_AH.zip")
    with zipfile.ZipFile(voice_archive) as archive:
        member = next(n for n in archive.namelist() if Path(n).stem == voice_id)
        (DEST / f"voice_{class_name}.wav").write_bytes(archive.read(member))

    gait_id = latest["Gait analysis"]["sampleId"]
    shutil.copy2(ROOT / "research_data" / "gaitpdb" / f"{gait_id}.txt", DEST / f"gait_{class_name}.txt")

print(f"Created six real dataset inputs in {DEST}")
