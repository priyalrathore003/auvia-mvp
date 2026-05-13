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
    Convert Hz boundaries to STFT bin indices, correctly for any sample rate.
    Fixes the hardcoded-bin bug that breaks 16kHz / 22kHz voice files.
    """
    bin_width = sr / n_fft
    low_bin  = int(np.floor(low_hz  / bin_width))
    high_bin = int(np.ceil (high_hz / bin_width))
    return low_bin, high_bin


def apply_neural_timbre_correction(audio_bytes: bytes) -> bytes:
    """
    Core spectral shaping — runs entirely in memory via BytesIO.
    No /tmp writes, no file-handle cleanup needed.
    """
    # --- Load ---
    input_buffer = io.BytesIO(audio_bytes)
    # sr=None preserves original sample rate; mono=True keeps memory predictable
    y, sr = librosa.load(input_buffer, sr=None, mono=True)
    logger.info(f"Loaded audio: {len(y)/sr:.2f}s @ {sr}Hz")

    # --- Harmonic / Percussive separation ---
    y_harm = librosa.effects.harmonic(y, margin=3.0)

    # --- Spectral shaping ---
    N_FFT = 2048
    S = librosa.stft(y_harm, n_fft=N_FFT)
    S_mag, S_phase = librosa.magphase(S)

    # Boost 150–500 Hz (low-mid vocal warmth), calculated per actual sr
    low_bin, high_bin = _compute_freq_bins(sr, N_FFT, 150.0, 500.0)
    S_mag[low_bin:high_bin, :] *= 1.5

    y_enhanced = librosa.istft(S_mag * S_phase, n_fft=N_FFT)

    # --- Write to in-memory buffer ---
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
    Accepts a WAV upload, returns an enhanced WAV via StreamingResponse.
    All processing is in-memory; no disk I/O, no cleanup race conditions.
    """
    if not file.filename.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    try:
        audio_bytes = await file.read()
        logger.info(f"Received '{file.filename}' — {len(audio_bytes)/1024:.1f} KB")

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