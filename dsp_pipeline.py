"""
dsp_pipeline.py
---------------
Core DSP (Digital Signal Processing) pipeline for Auvia audio enhancement.

Provides the `apply_timbre_enhancement` function that performs two-stage
vocal enhancement: noise reduction + spectral shaping.
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

    Stage 1 — Noise Reduction
        Spectral gating via noisereduce to remove room noise, mic hiss,
        AC hum, and breath noise.

    Stage 2 — Spectral Shaping
        Single STFT pass that:
          - Warms low-mids (150–500 Hz × 1.4)
          - Lifts presence (2–5 kHz × 1.15)
          - Rolls off sub-bass rumble (< 80 Hz × 0.3)

    The result is peak-normalised to −1 dBFS and written as 16-bit PCM WAV.

    Args:
        input_bytes: Raw WAV file bytes.

    Returns:
        Processed WAV file bytes.

    Raises:
        ValueError: If the decoded audio is silent or corrupt.
    """
    import noisereduce as nr

    # Write input bytes to a temp file so soundfile can read it
    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_in.write(input_bytes)
    tmp_in.close()

    try:
        # Read audio
        audio, sr = sf.read(tmp_in.name)
    except Exception as exc:
        os.unlink(tmp_in.name)
        raise ValueError(f"Failed to decode audio: {exc}") from exc

    if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
        os.unlink(tmp_in.name)
        raise ValueError("Decoded audio is empty — possible corrupt input.")

    # Ensure 2D array for consistent processing
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    # ------------------------------------------------------------------
    # Stage 1 — Noise Reduction (spectral gating)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Stage 2 — Spectral Shaping (STFT-based EQ)
    # ------------------------------------------------------------------
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

        # Build gain mask
        gain = np.ones_like(freqs)

        # Sub-bass roll-off (< 80 Hz × 0.3)
        gain[freqs < 80] = 0.3

        # Low-mid warmth (150–500 Hz × 1.4)
        mask_low_mid = (freqs >= 150) & (freqs <= 500)
        gain[mask_low_mid] = 1.4

        # Presence lift (2–5 kHz × 1.15)
        mask_presence = (freqs >= 2000) & (freqs <= 5000)
        gain[mask_presence] = 1.15

        # Apply gain (broadcast across all frames)
        D_shaped = D * gain[:, np.newaxis]

        # ISTFT back to time domain
        shaped[:, ch] = istft(
            D_shaped,
            hop_length=hop_length,
            win_length=win_length,
            length=len(reduced[:, ch]),
        )

    # ------------------------------------------------------------------
    # Peak normalise to −1 dBFS
    # ------------------------------------------------------------------
    peak = np.max(np.abs(shaped))
    if peak > 0:
        target = 10 ** (-1.0 / 20)  # −1 dBFS linear
        shaped = shaped * (target / peak)

    # Clip to prevent overshoot
    shaped = np.clip(shaped, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Write output WAV to bytes
    # ------------------------------------------------------------------
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