"""
dsp_pipeline.py
---------------
Core DSP pipeline for Auvia MVP: noise reduction + spectral shaping.
"""

import logging
import os
import tempfile

import numpy as np
import soundfile as sf

logger = logging.getLogger("auvia_dsp")


def apply_timbre_enhancement(input_bytes: bytes) -> bytes:
    """
    Two-stage vocal enhancement pipeline.

    Stage 1 — Noise Reduction (noisereduce spectral gating)
    Stage 2 — Spectral Shaping (librosa STFT EQ)
    """
    import noisereduce as nr

    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_in.write(input_bytes)
    tmp_in.close()

    try:
        audio, sr = sf.read(tmp_in.name)
    except Exception as exc:
        os.unlink(tmp_in.name)
        raise ValueError(f"Failed to decode audio: {exc}") from exc

    if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
        os.unlink(tmp_in.name)
        raise ValueError("Decoded audio is empty — possible corrupt input.")

    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    reduced = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        reduced[:, ch] = nr.reduce_noise(
            y=audio[:, ch],
            sr=sr,
            prop_decrease=0.85,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_std_thresh_stationary=1.5,
        )

    from librosa.core import istft, stft

    n_fft = 2048
    hop_length = 512
    win_length = 2048
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    shaped = np.zeros_like(reduced)
    for ch in range(reduced.shape[1]):
        D = stft(
            reduced[:, ch],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        gain = np.ones_like(freqs)
        gain[freqs < 80] = 0.3
        gain[(freqs >= 150) & (freqs <= 500)] = 1.4
        gain[(freqs >= 2000) & (freqs <= 5000)] = 1.15

        D_shaped = D * gain[:, np.newaxis]
        shaped[:, ch] = istft(
            D_shaped,
            hop_length=hop_length,
            win_length=win_length,
            length=len(reduced[:, ch]),
        )

    peak = np.max(np.abs(shaped))
    if peak > 0:
        target = 10 ** (-1.0 / 20)
        shaped = shaped * (target / peak)

    shaped = np.clip(shaped, -1.0, 1.0)

    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_out.close()
    try:
        sf.write(tmp_out.name, shaped, sr, subtype="PCM_16")
        with open(tmp_out.name, "rb") as f:
            output_bytes = f.read()
    finally:
        os.unlink(tmp_out.name)
        os.unlink(tmp_in.name)

    return output_bytes
