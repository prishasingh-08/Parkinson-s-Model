"""Feature extraction for the public PhysioNet GaitPDB insole force format."""
from __future__ import annotations

from io import BytesIO
import numpy as np

SAMPLE_RATE = 100.0


def parse_gait_record(payload: bytes) -> np.ndarray:
    """Parse a GaitPDB TXT recording: time plus 16 foot sensors and 2 totals."""
    try:
        data = np.loadtxt(BytesIO(payload), dtype=np.float64)
    except Exception as exc:
        raise ValueError("Please upload a compatible GaitPDB sensor TXT file.") from exc
    if data.ndim != 2 or data.shape[1] != 19 or data.shape[0] < 500:
        raise ValueError("The gait recording must contain at least 5 seconds of 19-column insole sensor data.")
    if not np.isfinite(data).all():
        raise ValueError("The gait file contains invalid sensor values.")
    time = data[:, 0]
    if np.any(np.diff(time) <= 0):
        raise ValueError("The gait recording's time column must increase steadily.")
    return data[:, 1:]


def extract_gait_features(data: np.ndarray) -> np.ndarray:
    """Extract 152 deterministic statistics from 18 vertical force channels."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 18 or data.shape[0] < 500 or not np.isfinite(data).all():
        raise ValueError("Expected at least 5 seconds of 18-channel insole force data.")
    out: list[float] = []
    for j in range(18):
        x = data[:, j]
        centered = x - np.mean(x)
        out.extend([float(np.mean(x)), float(np.std(x)), float(np.percentile(x, 90) - np.percentile(x, 10)),
                    float(np.mean(np.abs(np.diff(x)))), float(np.mean(x > max(5.0, np.percentile(x, 75) * .5)))])
        power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
        freq = np.fft.rfftfreq(len(centered), d=1.0 / SAMPLE_RATE)
        if len(power) > 1:
            power, freq = power[1:], freq[1:]
        p = power / (power.sum() + 1e-12)
        out.extend([float(freq[int(np.argmax(power))]) if len(power) else 0.0,
                    float(-np.sum(p * np.log(p + 1e-12)))])
    # Use total-force channels to summarize stance and load balance.
    left, right = data[:, 16], data[:, 17]
    lm = np.linalg.norm(data[:, :8], axis=1)
    rm = np.linalg.norm(data[:, 8:16], axis=1)
    for x in (left, right, left + right, left - right, lm, rm):
        out.extend([float(np.mean(x)), float(np.std(x)), float(np.percentile(x, 90) - np.percentile(x, 10))])
    result = np.asarray(out, dtype=np.float64)
    if result.shape != (144,) or not np.isfinite(result).all():
        raise ValueError("Could not extract stable gait measurements from this recording.")
    return result
