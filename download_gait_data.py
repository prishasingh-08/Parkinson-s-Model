"""Fetch one usual-walk recording per participant from public PhysioNet GaitPDB."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
import re
import numpy as np

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "research_data" / "gaitpdb"
BASE = "https://physionet-open.s3.amazonaws.com/gaitpdb/1.0.0/"
LISTING_BASE = "https://physionet.org/files/gaitpdb/1.0.0/"
DEST.mkdir(parents=True, exist_ok=True)

with urlopen(LISTING_BASE, timeout=60) as response:
    listing = response.read().decode("utf-8", errors="replace")
names = sorted(set(re.findall(r'href="([A-Za-z]{4}\d{2}_01\.txt)"', listing)))
print(f"Downloading {len(names)} usual-walk recordings")

def fetch(name):
    path = DEST / name
    if path.exists():
        try:
            old = np.loadtxt(path, dtype=np.float64)
            if old.ndim == 2 and old.shape[1] == 19 and old.shape[0] >= 500 and np.isfinite(old).all() and np.all(np.diff(old[:, 0]) > 0):
                return name, path.stat().st_size
        except Exception:
            pass
    last_error = None
    for attempt in range(3):
        try:
            with urlopen(BASE + name, timeout=45) as response:
                payload = response.read()
            data = np.loadtxt(BytesIO(payload), dtype=np.float64)
            if data.ndim != 2 or data.shape[1] != 19 or data.shape[0] < 500 or not np.isfinite(data).all() or not np.all(np.diff(data[:, 0]) > 0):
                raise RuntimeError(f"Invalid GaitPDB shape for {name}: {data.shape}")
            temp_path = path.with_suffix(path.suffix + ".download")
            temp_path.write_bytes(payload)
            temp_path.replace(path)
            return name, len(payload)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not fetch a valid {name}: {last_error}")

done = []
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(fetch, name) for name in names]
    for future in as_completed(futures):
        name, size = future.result()
        done.append(name)
        if len(done) % 20 == 0 or len(done) == len(names):
            print(f"Downloaded {len(done)}/{len(names)}")
print(f"Finished: {len(done)} files, {sum((DEST / n).stat().st_size for n in done)/1e6:.1f} MB")
