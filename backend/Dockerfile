FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models/hf TORCH_HOME=/models/torch DOCTR_CACHE_DIR=/models/doctr
WORKDIR /srv

# libgl/glib are needed by opencv (used by doctr); build tools for any wheels that need compiling
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch keeps the image small; drop --index-url for a CUDA build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
