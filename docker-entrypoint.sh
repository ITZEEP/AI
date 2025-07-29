#!/bin/bash
set -e

echo "🚀 Starting AI OCR Service with Auto-Initialization..."

# Convert relative path to absolute path if needed
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    if [[ ! "$GOOGLE_APPLICATION_CREDENTIALS" = /* ]]; then
        export GOOGLE_APPLICATION_CREDENTIALS="/app/$GOOGLE_APPLICATION_CREDENTIALS"
        echo "✅ Converted credentials path to: $GOOGLE_APPLICATION_CREDENTIALS"
    fi
    
    if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        echo "⚠️ Google Cloud credentials file not found at $GOOGLE_APPLICATION_CREDENTIALS"
        echo "   Please ensure the credentials file exists for OCR functionality."
    else
        echo "✅ Google Cloud credentials found"
    fi
fi

# Clean up old temporary files
find /app/temp -type f -mtime +1 -delete 2>/dev/null || true

# Check if vectorstore already exists
if [ -d "/app/data/vectorstore" ] && [ -n "$(ls -A /app/data/vectorstore 2>/dev/null)" ]; then
    echo "✅ Law vectorstore already exists"
else
    echo "🔧 Initializing vectorstore with all packages..."
    
    # Ensure all required packages are available (already in requirements.txt)
    echo "📦 All packages installed during Docker build"
    
    # Check if law_docs directory has PDF files
    if [ -d "/app/data/law_docs" ] && [ "$(ls -A /app/data/law_docs/*.pdf 2>/dev/null)" ]; then
        echo "📄 Found law documents, initializing vectorstore..."
        python init_vectorstore.py && echo "✅ Vectorstore initialized successfully" || echo "⚠️ Vectorstore initialization failed, continuing without legal analysis"
    else
        echo "📁 No law documents found, creating directory structure..."
        mkdir -p /app/data/law_docs
        cat > /app/data/law_docs/README.txt << 'EOF'
법령 PDF 파일 추가 방법:
1. 이 디렉토리에 법령 PDF 파일을 복사하세요
2. docker-compose restart로 서비스를 재시작하세요
3. 벡터스토어가 자동으로 초기화됩니다

예시 파일명:
- 주택임대차보호법.pdf
- 민법_임대차편.pdf
- 부동산등기법.pdf
EOF
        echo "📋 Please add PDF law documents to /app/data/law_docs/ and restart to enable legal analysis"
    fi
fi

# Disable warnings
export CHROMA_TELEMETRY_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "✅ Environment setup completed - Starting service..."

# Execute the main command
exec "$@"