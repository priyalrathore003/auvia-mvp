import io
import asyncio
import logging
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auvia_engine")

app = FastAPI(title="Auvia Audio Engine - V1 Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _compute_freq_bins(sr: int, n_fft: int, low_hz: float, high_hz: float):
    """Convert Hz boundaries to STFT bin indices for any sample rate."""
    bin_width = sr / n_fft
    low_bin  = int(np.floor(low_hz  / bin_width))
    high_bin = int(np.ceil (high_hz / bin_width))
    return low_bin, high_bin


def apply_timbre_enhancement(audio_bytes: bytes) -> bytes:
    """
    Spectral shaping via single STFT pass.

    Removed librosa.effects.harmonic — it ran a second internal STFT,
    doubled processing time on Render's 0.5 CPU, and caused single-frequency
    resonance artifacts (the 'beep') on compressed MP3 inputs.

    Processing chain: load → STFT → spectral boost → ISTFT → normalise → write
    """
    input_buffer = io.BytesIO(audio_bytes)

    # 22050 Hz: half the memory of 44.1kHz, fine for vocal processing
    y, sr = librosa.load(input_buffer, sr=22050, mono=True)
    logger.info(f"Loaded: {len(y)/sr:.2f}s @ {sr}Hz | peak={np.max(np.abs(y)):.4f}")

    # Explicit hop_length on both stft + istft — must match or reconstruction breaks
    N_FFT     = 1024
    HOP       = 256   # N_FFT // 4 — standard for perfect reconstruction

    S         = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    del y

    S_mag, S_phase = librosa.magphase(S)
    del S

    # ── Spectral shaping ──────────────────────────────────────────────────────
    # Low-mid warmth boost: 150–500 Hz
    lo, hi = _compute_freq_bins(sr, N_FFT, 150.0, 500.0)
    S_mag[lo:hi, :] *= 1.4

    # Gentle high-mid presence lift: 2kHz–5kHz (air/clarity)
    lo2, hi2 = _compute_freq_bins(sr, N_FFT, 2000.0, 5000.0)
    S_mag[lo2:hi2, :] *= 1.15

    # Soft low-end cut below 80 Hz (remove rumble/proximity buildup)
    lo_cut, _ = _compute_freq_bins(sr, N_FFT, 0.0, 80.0)
    S_mag[:lo_cut, :] *= 0.3
    # ─────────────────────────────────────────────────────────────────────────

    y_enhanced = librosa.istft(S_mag * S_phase, n_fft=N_FFT, hop_length=HOP)
    del S_mag, S_phase

    # Peak normalise to -1 dBFS
    peak = np.max(np.abs(y_enhanced))
    logger.info(f"Post-ISTFT peak: {peak:.6f}")
    if peak < 1e-6:
        raise ValueError("Processing produced silence — input may be corrupt.")
    y_enhanced = y_enhanced / peak * 0.891

    output_buffer = io.BytesIO()
    sf.write(output_buffer, y_enhanced, sr, format="WAV", subtype="PCM_16")
    output_buffer.seek(0)

    logger.info("Processing complete.")
    return output_buffer.read()


@app.get("/")
async def health():
    return {"status": "Auvia Engine is live"}


@app.post("/process-vocal/")
async def process_vocal(file: UploadFile = File(...)):
    ALLOWED = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    MAX_MB  = 10

    if not file.filename.lower().endswith(ALLOWED):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Accepted: {', '.join(ALLOWED)}"
        )

    audio_bytes = await file.read()

    if len(audio_bytes) > MAX_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {MAX_MB}MB on current plan."
        )

    logger.info(f"Received '{file.filename}' — {len(audio_bytes)/1024:.1f} KB")

    try:
        processed_bytes = await asyncio.to_thread(apply_timbre_enhancement, audio_bytes)

        return StreamingResponse(
            io.BytesIO(processed_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="auvia_enhanced.wav"',
                "Content-Length": str(len(processed_bytes)),
            },
        )

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))