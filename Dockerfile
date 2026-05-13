FROM python:3.11-slim

   # Install system audio dependencies
   RUN apt-get update && apt-get install -y \
       libsndfile1 \
       ffmpeg \
       && rm -rf /var/lib/apt/lists/*

   WORKDIR /app
   COPY . .

   RUN pip install --no-cache-dir -r requirements.txt

   # Start the FastAPI engine on Render's required port
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]