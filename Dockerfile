# Multi-stage build for smaller image size
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

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
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

# Copy and set entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Use entrypoint for initialization
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Run the application with workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]