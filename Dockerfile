FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends     tesseract-ocr     tesseract-ocr-rus     tesseract-ocr-eng     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000
CMD ["sh", "-c", "uvicorn bot.main:app --host 0.0.0.0 --port ${PORT}"]
