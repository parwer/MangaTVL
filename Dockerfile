# MangaTVL API — CPU image (default).
# For GPU, use Dockerfile.gpu instead.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    DEVICE=cpu

WORKDIR /app

# System libs required by opencv-python (libGL/libxcb/X) — ultralytics pulls in
# the non-headless opencv, so these are needed for `import cv2` to work.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install deps first for better layer caching (CPU torch comes from PyPI default index)
COPY requirement.txt .
RUN pip install --upgrade pip && pip install -r requirement.txt

# App code + bundled assets (font + best_diplom.pt model are copied in)
COPY . .

# Pre-download EasyOCR English models so the container is self-contained (no first-request download)
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
