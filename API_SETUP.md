# Google Cloud Vision API 설정 가이드

## 1. Google Cloud 프로젝트 설정

### 1.1 Google Cloud Console 접속
- https://console.cloud.google.com/ 접속
- Google 계정으로 로그인

### 1.2 프로젝트 생성
- 상단의 프로젝트 선택 드롭다운 클릭
- "새 프로젝트" 클릭
- 프로젝트 이름 입력 (예: "itzip-ocr")
- "만들기" 클릭

### 1.3 Cloud Vision API 사용 설정
1. 좌측 메뉴에서 "API 및 서비스" → "라이브러리" 클릭
2. "Cloud Vision API" 검색
3. "Cloud Vision API" 클릭 후 "사용" 버튼 클릭

## 2. 서비스 계정 생성 및 키 다운로드

### 2.1 서비스 계정 생성
1. 좌측 메뉴에서 "API 및 서비스" → "사용자 인증 정보" 클릭
2. 상단의 "+ 사용자 인증 정보 만들기" → "서비스 계정" 클릭
3. 서비스 계정 세부정보 입력:
   - 서비스 계정 이름: `itzip-ocr-service`
   - 서비스 계정 ID: 자동 생성됨
   - "만들고 계속하기" 클릭
4. 역할 선택:
   - "Cloud Vision API 사용자" 선택
   - "계속" 클릭
5. "완료" 클릭

### 2.2 JSON 키 다운로드
1. 생성된 서비스 계정 클릭
2. "키" 탭 클릭
3. "키 추가" → "새 키 만들기" 클릭
4. "JSON" 선택 후 "만들기" 클릭
5. JSON 파일이 자동으로 다운로드됨

## 3. 프로젝트 설정

### 3.1 JSON 키 파일 배치
다운로드한 JSON 파일을 안전한 위치에 저장:
```
D:\itzip\AI-develop\credentials\
└── your-project-key.json
```

### 3.2 .env 파일 설정
`D:\itzip\AI-develop\.env` 파일을 열고 수정:

```env
# Google Cloud Vision API 설정
GOOGLE_APPLICATION_CREDENTIALS=D:/itzip/AI-develop/credentials/your-project-key.json

# Mock 모드 비활성화 (실제 API 사용)
MOCK_OCR_MODE=false
```

### 3.3 .gitignore에 추가
보안을 위해 credentials 폴더를 .gitignore에 추가:

```gitignore
# Google Cloud credentials
credentials/
*.json
.env
```

## 4. 테스트

서버 재시작 후 테스트:

```bash
# 서버 시작
cd AI-develop
python main.py

# 다른 터미널에서 테스트
curl http://localhost:8000/health
```

응답에서 `"building_parser_available": true`가 나오면 성공!

## 5. 요금 정보

Google Cloud Vision API 요금:
- 매월 첫 1,000개 요청: 무료
- 이후: 1,000개당 $1.50

신규 가입 시 $300 크레딧 제공 (90일간 사용 가능)

## 6. Docker 환경에서 실행

### 6.1 Docker 빌드 및 실행

```bash
# 1. 프로젝트 디렉토리로 이동
cd D:\itzip\AI-develop

# 2. .env 파일 생성 (아직 없다면)
copy .env.example .env

# 3. .env 파일 수정 - Google API 키 설정
# GOOGLE_APPLICATION_CREDENTIALS=credentials/google-vision-key.json
# GOOGLE_API_KEY=your-google-api-key-here

# 4. Docker 이미지 빌드
docker-compose build

# 5. Docker 컨테이너 실행
docker-compose up -d

# 6. 로그 확인
docker-compose logs -f ai-service
```

### 6.2 Docker 환경 변수

Docker에서는 다음과 같이 경로가 자동 변환됩니다:
- 로컬: `credentials/google-vision-key.json`
- Docker: `/app/credentials/google-vision-key.json`

### 6.3 Docker 볼륨 마운트

docker-compose.yml에서 다음 볼륨이 마운트됩니다:
- `./credentials:/app/credentials:ro` - Google Cloud 인증 파일 (읽기 전용)
- `./logs:/app/logs` - 로그 파일 저장
- `./data:/app/data` - 벡터스토어 및 법령 문서
- `./temp:/app/temp` - 임시 파일 처리

### 6.4 Docker 명령어

```bash
# 컨테이너 시작
docker-compose up -d

# 컨테이너 중지
docker-compose down

# 로그 확인
docker-compose logs -f

# 컨테이너 재시작
docker-compose restart

# 컨테이너 상태 확인
docker-compose ps
```

## 주의사항

- JSON 키 파일은 절대 Git에 커밋하지 마세요
- 프로덕션 환경에서는 환경 변수나 Secret Manager 사용 권장
- API 사용량은 Google Cloud Console에서 모니터링 가능
- Docker 환경에서는 상대 경로 사용을 권장합니다