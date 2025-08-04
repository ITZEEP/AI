# Multi-stage build for smaller image size and better caching
FROM python:3.12-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and PDF processing
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    # SQLite3 (for ChromaDB)
    sqlite3 \
    libsqlite3-dev \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # PDF and image processing
    libgdal-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-kor \
    # Additional dependencies for PyMuPDF
    libgtk-3-0 \
    libnotify-dev \
    libsdl-pango-dev \
    libwebp-dev \
    zlib1g-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    libpng-dev \
    libtiff-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt .

# Upgrade pip in a separate layer
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies in a separate layer
# This layer is only rebuilt when requirements.txt changes
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code last to maximize cache usage
COPY . .

# Create directories for credentials and logs
RUN mkdir -p credentials logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8000
ENV HOST=0.0.0.0
# ChromaDB SQLite check bypass (for older SQLite versions)
ENV CHROMA_SERVER_NOFILE=1
ENV ALLOW_RESET=TRUE

# Expose port
EXPOSE 8000

# Create necessary directories
RUN mkdir -p temp data/vectorstore data/law_docs

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Set environment variables for ChromaDB and warnings
ENV CHROMA_TELEMETRY_DISABLED=true
ENV TOKENIZERS_PARALLELISM=false

# Default number of workers
ENV WORKERS=3

# Default command (can be overridden by docker-compose)
CMD ["python", "-m", "app.main"]