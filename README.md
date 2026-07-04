# Auvia MVP

Minimal DSP audio enhancement API for vocal cleanup.

## What it does

Two-stage pipeline in `dsp_pipeline.py`:

1. **Noise reduction** — `noisereduce` spectral gating
2. **Spectral shaping** — librosa STFT EQ (sub-bass roll-off, low-mid warmth, presence lift)

## Run locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

## Endpoint

`POST /process-vocal/` — upload a WAV/MP3/FLAC/OGG/M4A file, receive enhanced WAV.

```bash
curl -X POST http://localhost:8000/process-vocal/ \
  -F "file=@input.wav" \
  -o enhanced.wav
```

## Docker

```bash
docker build -t auvia-mvp .
docker run -p 10000:10000 auvia-mvp
```
