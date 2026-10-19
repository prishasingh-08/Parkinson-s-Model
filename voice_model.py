"""Feature extraction for sustained /a/ voice recordings (research prototype)."""
from __future__ import annotations

from io import BytesIO
from math import gcd

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _frames(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))

    count = 1 + (len(signal) - frame_len) // hop
    indices = np.arange(frame_len)[None, :] + hop * np.arange(count)[:, None]
    return signal[indices]


def extract_voice_features(payload: bytes) -> np.ndarray:
    """Extract acoustic features from a WAV recording of a sustained /a/."""
    try:
        sample_rate, audio = wavfile.read(BytesIO(payload))
    except Exception as exc:
        raise ValueError("Please upload a valid WAV voice recording.") from exc

    original_dtype = audio.dtype

    if audio.ndim == 2:
        audio = audio.astype(np.float64).mean(axis=1)

    if audio.ndim != 1 or sample_rate < 4000 or len(audio) < sample_rate // 2:
        raise ValueError("The voice recording is too short or has an unsupported format.")

    audio = np.asarray(audio, dtype=np.float64)

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        audio /= max(abs(info.min), info.max)

    audio = np.nan_to_num(audio)

    # Check the original signal before normalization. Otherwise, tiny
    # microphone noise in a silent recording could be amplified into a
    # full-volume signal and receive a misleading model result.
    input_peak = float(np.max(np.abs(audio)))
    input_rms = float(np.sqrt(np.mean(audio * audio) + 1e-15))

    if input_peak < 0.003 or input_rms < 0.0006:
        raise ValueError(
            "We couldn't hear a clear voice. Record a steady 'ah' for about "
            "4 seconds, closer to the microphone, then try again."
        )

    # Match the telephone-quality sample rate used by the training recordings.
    if sample_rate != 8000:
        ratio = gcd(int(sample_rate), 8000)
        audio = resample_poly(audio, 8000 // ratio, int(sample_rate) // ratio)
        sample_rate = 8000

    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        raise ValueError("No usable voice signal was detected in the recording.")

    audio /= peak

    # Remove DC drift and retain the speech band.
    high = min(3500.0, sample_rate * 0.47)
    audio = sosfiltfilt(
        butter(3, [60.0, high], btype="bandpass", fs=sample_rate, output="sos"),
        audio,
    )

    frame_len = max(128, int(round(0.040 * sample_rate)))
    hop = max(64, int(round(0.010 * sample_rate)))
    frames = _frames(audio, frame_len, hop)
    window = np.hamming(frame_len)

    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    active = rms > max(np.percentile(rms, 20) * 1.05, 0.002)

    if active.sum() < 5:
        raise ValueError(
            "The recording is too quiet. Please record a clear, steady vowel /a/."
        )

    frames = frames[active]
    rms = rms[active]

    nfft = max(512, 2 ** int(np.ceil(np.log2(frame_len))))
    spectrum = np.abs(np.fft.rfft(frames * window, n=nfft, axis=1)) + 1e-10
    power = spectrum**2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)

    spectral_sum = spectrum.sum(axis=1)
    centroid = (spectrum * freqs).sum(axis=1) / spectral_sum
    bandwidth = np.sqrt(
        (spectrum * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1)
        / spectral_sum
    )

    roll_index = np.argmax(
        np.cumsum(power, axis=1) >= 0.85 * power.sum(axis=1)[:, None],
        axis=1,
    )
    rolloff = freqs[roll_index]
    zcr = np.mean(
        np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]),
        axis=1,
    )

    # Cepstral coefficients summarize the vocal-tract spectrum.
    mel_points = np.linspace(
        _hz_to_mel(np.array([60.0]))[0],
        _hz_to_mel(np.array([high]))[0],
        28,
    )
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((26, len(freqs)), dtype=np.float64)
    for m in range(1, 27):
        left, center, right = bins[m - 1 : m + 2]
        center = max(center, left + 1)
        right = max(right, center + 1)

        for k in range(max(left, 0), min(center, len(freqs))):
            filterbank[m - 1, k] = (k - left) / (center - left)

        for k in range(max(center, 0), min(right, len(freqs))):
            filterbank[m - 1, k] = (right - k) / (right - center)

    mel_energy = power @ filterbank.T
    cepstra = dct(
        np.log(mel_energy + 1e-8),
        type=2,
        axis=1,
        norm="ortho",
    )[:, 1:13]

    # Estimate pitch and periodicity.
    pitches = []
    periods = []
    min_lag = max(2, int(sample_rate / 400.0))
    max_lag = min(frame_len - 2, int(sample_rate / 70.0))

    for frame in frames:
        x = frame * window
        x = x - x.mean()
        corr = np.correlate(x, x, mode="full")[len(x) - 1 :]

        if corr[0] <= 1e-8 or max_lag <= min_lag:
            continue

        corr /= corr[0]
        lag = min_lag + int(np.argmax(corr[min_lag : max_lag + 1]))

        if corr[lag] < 0.30:
            continue

        delta = 0.0
        if 1 <= lag < len(corr) - 1:
            a, b, c = corr[lag - 1 : lag + 2]
            denom = a - 2 * b + c
            if abs(denom) > 1e-10:
                delta = 0.5 * (a - c) / denom

        period = lag + delta
        periods.append(period / sample_rate)
        pitches.append(sample_rate / period)

    pitches = np.asarray(pitches)
    periods = np.asarray(periods)

    if len(pitches) < 3:
        pitch_stats = [0.0] * 5
        jitter = 0.0
    else:
        pitch_stats = [
            float(np.median(pitches)),
            float(np.std(pitches)),
            float(np.percentile(pitches, 10)),
            float(np.percentile(pitches, 90)),
            float(len(pitches) / len(frames)),
        ]
        jitter = (
            float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-9))
            if len(periods) > 1
            else 0.0
        )

    features = [
        float(np.log(np.mean(rms) + 1e-8)),
        float(np.std(rms)),
        float(np.percentile(rms, 90) - np.percentile(rms, 10)),
        float(np.mean(centroid)),
        float(np.std(centroid)),
        float(np.mean(bandwidth)),
        float(np.mean(rolloff)),
        float(np.std(rolloff)),
        float(np.mean(zcr)),
        float(np.std(zcr)),
        float(np.mean(np.diff(cepstra, axis=0) ** 2)),
        jitter,
        *pitch_stats,
    ]

    features.extend(np.mean(cepstra, axis=0).tolist())
    features.extend(np.std(cepstra, axis=0).tolist())

    result = np.asarray(features, dtype=np.float64)

    if result.shape != (41,) or not np.isfinite(result).all():
        raise ValueError("Could not extract stable voice measurements from this file.")

    return result