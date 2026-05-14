import io
import asyncio
import logging
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auvia_engine")

app = FastAPI(title="Auvia Audio Engine - V1 Core")


def _compute_freq_bins(sr: int, n_fft: int, low_hz: float, high_hz: float):
    """
    Convert Hz boundaries to STFT bin indices for any sample rate.
    Fixes the hardcoded-bin bug that breaks 16kHz / 22kHz voice files.
    """
    bin_width = sr / n_fft
    low_bin  = int(np.floor(low_hz  / bin_width))
    high_bin = int(np.ceil (high_hz / bin_width))
    return low_bin, high_bin


def apply_neural_timbre_correction(audio_bytes: bytes) -> bytes:
    """
    Core spectral shaping — runs entirely in memory via BytesIO.
    Memory-optimised: intermediates deleted immediately after use
    to stay within 512MB on Render Starter.
    """
    input_buffer = io.BytesIO(audio_bytes)

    # sr=22050 halves memory vs native 44.1kHz — fine for vocal processing
    y, sr = librosa.load(input_buffer, sr=22050, mono=True)
    logger.info(f"Loaded: {len(y)/sr:.2f}s @ {sr}Hz | peak={np.max(np.abs(y)):.4f}")

    # Harmonic / percussive separation
    y_harm = librosa.effects.harmonic(y, margin=3.0)
    del y  # free original signal immediately

    # n_fft=1024 quarters STFT matrix size vs 2048
    N_FFT = 1024
    S = librosa.stft(y_harm, n_fft=N_FFT)
    del y_harm  # free harmonic signal immediately

    S_mag, S_phase = librosa.magphase(S)
    del S  # free full complex STFT immediately

    # Boost 150-500 Hz (low-mid vocal warmth), calculated per actual sr
    low_bin, high_bin = _compute_freq_bins(sr, N_FFT, 150.0, 500.0)
    S_mag[low_bin:high_bin, :] *= 1.5

    y_enhanced = librosa.istft(S_mag * S_phase, n_fft=N_FFT)
    del S_mag, S_phase  # free before normalise + write

    # Peak normalise to -1dBFS
    # Without this, ISTFT output can be near-zero and PCM_16 writes silence
    peak = np.max(np.abs(y_enhanced))
    logger.info(f"Post-ISTFT peak: {peak:.6f}")
    if peak < 1e-6:
        raise ValueError("Processing produced silence — input may be corrupt or empty.")
    y_enhanced = y_enhanced / peak * 0.891  # 0.891 = -1 dBFS headroom

    output_buffer = io.BytesIO()
    sf.write(output_buffer, y_enhanced, sr, format="WAV", subtype="PCM_16")
    output_buffer.seek(0)

    logger.info("Processing complete — returning enhanced audio.")
    return output_buffer.read()


@app.get("/")
async def health():
    return {"status": "Auvia Engine is live"}


@app.post("/process-vocal/")
async def process_vocal(file: UploadFile = File(...)):
    """
    Accepts WAV / MP3 / FLAC / OGG / M4A upload.
    Returns enhanced WAV via StreamingResponse.
    All processing is in-memory — no disk I/O, no cleanup race conditions.
    """
    ALLOWED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    MAX_FILE_MB = 10

    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Accepted: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    audio_bytes = await file.read()

    if len(audio_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size on current plan: {MAX_FILE_MB}MB."
        )

    logger.info(f"Received '{file.filename}' — {len(audio_bytes)/1024:.1f} KB")

    try:
        processed_bytes = await asyncio.to_thread(
            apply_neural_timbre_correction, audio_bytes
        )

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