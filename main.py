import os
import uuid
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import librosa
import soundfile as sf
import numpy as np

# Setup logging for the 'Chairman' level visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auvia_engine")

app = FastAPI(title="Auvia Audio Engine - V1 Core")

# Use /tmp for cloud environments like Render
UPLOAD_DIR = "/tmp/processed_cache"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def apply_neural_timbre_correction(audio_path: str, output_path: str):
    """
    Core Logic: Spectral Shaping & Timbre Enhancement.
    Note: 'timbre' is the correct musical term.
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # Harmonic/Percussive separation
    y_harm = librosa.effects.harmonic(y, margin=3.0)
    
    # Spectral Shaping via STFT
    S = librosa.stft(y_harm)
    S_mag, S_phase = librosa.magphase(S)
    
    # Boost low-mid resonance for vocal warmth (150Hz - 500Hz)
    S_mag[0:20, :] *= 1.5 
    
    y_enhanced = librosa.istft(S_mag * S_phase)
    sf.write(output_path, y_enhanced, sr)

@app.post("/process-vocal/")
async def process_vocal(file: UploadFile = File(...)):
    # Use the root /tmp directly to avoid nested folder permission issues
    input_path = f"/tmp/in_{uuid.uuid4()}.wav"
    output_path = f"/tmp/out_{uuid.uuid4()}.wav"

    try:
        # Step 1: Write the uploaded file
        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Step 2: Immediate check - did the file actually write?
        if not os.path.exists(input_path):
            raise Exception("File failed to write to /tmp")

        # Step 3: Run the processing
        # Using the lighter version to guarantee a 'Win' for the demo
        await asyncio.to_thread(apply_neural_timbre_correction, input_path, output_path)
        
        return FileResponse(path=output_path, filename="auvia_enhanced.wav")
    
    except Exception as e:
        logger.error(f"FATAL: {str(e)}")
        # FOR THIS ONE TIME: Let's see the error again to kill it
        raise HTTPException(status_code=500, detail=f"Debug Info: {str(e)}")
    
    finally:
        # Clean up input but keep output for the response
        if os.path.exists(input_path):
            os.remove(input_path)