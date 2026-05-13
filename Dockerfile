FROM python:3.11-slim

# Install system dependencies + setuptools at the OS level
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Force upgrade pip and install requirements
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install setuptools  # Triple-redundancy install

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]