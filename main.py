import asyncio
import io
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from dsp_pipeline import apply_timbre_enhancement

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auvia_mvp")

app = FastAPI(title="Auvia MVP — DSP Audio Enhancement")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health():
    return {"status": "Auvia MVP is live"}


@app.post("/process-vocal/")
async def process_vocal(file: UploadFile = File(...)):
    allowed = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    max_mb = 10

    if not file.filename.lower().endswith(allowed):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Accepted: {', '.join(allowed)}",
        )

    audio_bytes = await file.read()

    if len(audio_bytes) > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {max_mb}MB.",
        )

    logger.info("Received '%s' — %.1f KB", file.filename, len(audio_bytes) / 1024)

    try:
        processed_bytes = await asyncio.to_thread(
            apply_timbre_enhancement, audio_bytes
        )
    except Exception as exc:
        logger.error("Processing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return StreamingResponse(
        io.BytesIO(processed_bytes),
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="auvia_enhanced.wav"',
            "Content-Length": str(len(processed_bytes)),
        },
    )
