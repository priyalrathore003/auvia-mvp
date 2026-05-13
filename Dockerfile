FROM python:3.11-slim
 
# ── System deps ───────────────────────────────────────────────────────────────
# libsndfile1: required by soundfile at runtime (the actual C binding)
# ffmpeg:      required by librosa for mp3/ogg decode (audioread backend)
# No python3-setuptools here — let pip manage setuptools to avoid version conflicts
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app
 
# ── Python deps ───────────────────────────────────────────────────────────────
# CRITICAL ORDER: upgrade pip + setuptools BEFORE installing requirements.
# librosa imports pkg_resources (part of setuptools) at module load time.
# If setuptools is missing when the worker boots, every request crashes with
# "No module named 'pkg_resources'" — which is exactly the bug we're fixing.
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
 
# ── App code (separate layer so code changes don't reinstall deps) ─────────────
COPY . .
 
# Render sets PORT env var; default 10000 for local docker run
ENV PORT=10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]