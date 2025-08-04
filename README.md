# 잇집 AI

FastAPI 기반 부동산 문서 OCR 및 사기 위험도 분석 서비스

## 🚀 원클릭 시작

```bash
# 전체 서비스 빌드 및 시작 (모든 기능 자동 활성화)
docker-compose up --build
```

**끝!** 이제 http://localhost:8000 에서 모든 기능을 사용할 수 있습니다.

### 제공 기능

- ✅ 등기부등본 OCR 분석
- ✅ 건축물대장 OCR 분석
- ✅ 계약서 특약사항 추출
- ✅ 부동산 사기 위험도 분석 (전세/월세 지원)
- ✅ 계약서 법령 적법성 검증
- ✅ **벡터스토어 자동 설정** (법령 문서 추가 시 자동 활성화)

### 고급 기능 활성화 (선택사항)

법령 기반 상세 분석을 원한다면 PDF 법령 문서만 추가하세요:

```bash
# 1. 법령 PDF 파일 추가
mkdir -p data/law_docs
cp 주택임대차보호법.pdf data/law_docs/

# 2. 서비스 재시작
docker-compose restart
```

추가 기능:

- ✅ 법령 기반 위험도 분석
- ✅ 관련 법조문 자동 검색
- ✅ AI 법령 해석

## 📚 API 사용법

### 주요 엔드포인트

```bash
# 건강 상태 확인
curl https://ai.ariogi.kr/api/health

# 등기부등본 파싱
curl -X POST https://ai.ariogi.kr/api/parse/register \
  -F "file=@등기부등본.pdf"

# 건축물대장 파싱
curl -X POST https://ai.ariogi.kr/api/parse/building \
  -F "file=@건축물대장.pdf"

# 계약서 특약사항 추출
curl -X POST https://ai.ariogi.kr/api/parse/contract \
  -F "file=@임대차계약서.pdf"

# 위험도 분석 (전세)
curl -X POST https://ai.ariogi.kr/api/analyze/risk \
  -H "Content-Type: application/json" \
  -d '{
    "userId": 123,
    "userType": "tenant",
    "homeId": 456,
    "address": "서울특별시 강남구 테헤란로 123",
    "propertyPrice": 500000000,
    "leaseType": "JEONSE",
    "registryDocument": {...},
    "buildingDocument": {...},
    "registeredUserName": "김철수",
    "residenceType": "APARTMENT"
  }'

# 위험도 분석 (월세)
curl -X POST https://ai.ariogi.kr/api/analyze/risk \
  -H "Content-Type: application/json" \
  -d '{
    "userId": 123,
    "userType": "tenant",
    "homeId": 456,
    "address": "서울특별시 강남구 테헤란로 123",
    "propertyPrice": 100000000,
    "monthlyRent": 1500000,
    "leaseType": "WOLSE",
    "registryDocument": {...},
    "buildingDocument": {...},
    "registeredUserName": "김철수",
    "residenceType": "APARTMENT"
  }'

# 계약서 법령 검증
curl -X POST https://ai.ariogi.kr/api/contract/validate \
  -H "Content-Type: application/json" \
  -d @contract_validation_request.json
```

## 🚀 배포

### 자동 배포

- `main` 브랜치 → Production (ai.ariogi.kr)
- `develop` 브랜치 → Development (ai.dev.ariogi.kr)

### 수동 배포

```bash
# Docker 이미지 빌드
docker build -t itzip-ai:latest .

# 컨테이너 실행
docker run -d \
  --name itzip-ai-service \
  -p 8000:8000 \
  -v ./credentials:/app/credentials:ro \
  -v ./logs:/app/logs \
  -v ./data:/app/data \
  --env-file .env \
  itzip-ai:latest
```

## 🔧 문제 해결

### 법령 분석 기능 추가 방법

법령 기반 분석 기능을 사용하려면:

```bash
# 1. 법령 PDF 파일을 추가
mkdir -p data/law_docs
cp your_law_document.pdf data/law_docs/

# 2. 서비스 재시작 (자동 벡터스토어 초기화)
docker-compose restart
```

### 로그 확인

```bash
# 서비스 로그 확인
docker-compose logs -f

# 컨테이너 내부 확인
docker exec -it itzip-ai-service ls -la data/
```

### commit convention

✨ feat: 기능 추가, 삭제, 변경

🐛 fix: 버그, 오류 수정

📝 docs: readme.md, json 파일 등 수정, 라이브러리 설치 (문서 관련, 코드 수정 없음)

💄 style: CSS 등 사용자 UI 디자인 변경 (제품 코드 수정 발생, 코드 형식, 정렬, 주석 등의 변경)

♻ refactor: 코드 리팩토링

🧪 test: 테스트 코드 추가, 삭제, 변경 등 (코드 수정 없음, 테스트 코드에 관련된 모든 변경에 해당)

🔧 config: npm 모듈 설치 등

🌱 chore: (코드의 수정 없이) 설정 변경

## 구성

langchain/retriever.py 구성도

<img width="323" height="212" alt="image" src="https://github.com/user-attachments/assets/35e99767-fdc7-4e73-8eaa-e8d25cb0cdfb" />
