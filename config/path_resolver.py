import os
from pathlib import Path


def resolve_credentials_path(relative_path: str) -> str:
    """
    상대 경로를 절대 경로로 변환하는 헬퍼 함수
    Docker 환경과 로컬 환경 모두에서 작동
    
    Args:
        relative_path: 상대 경로 (예: "credentials/google-vision-key.json")
        
    Returns:
        절대 경로 문자열
    """
    # 이미 절대 경로인 경우 그대로 반환
    if os.path.isabs(relative_path):
        return relative_path
    
    # 프로젝트 루트 디렉토리 찾기
    # Docker 환경에서는 /app, 로컬에서는 현재 디렉토리 기준
    if os.path.exists('/app'):
        # Docker 환경
        base_path = Path('/app')
    else:
        # 로컬 환경 - 이 파일의 위치에서 프로젝트 루트 찾기
        base_path = Path(__file__).parent.parent
    
    # 상대 경로를 절대 경로로 변환
    absolute_path = base_path / relative_path
    
    return str(absolute_path.resolve())


def get_google_credentials_path() -> str:
    """
    환경 변수에서 Google Cloud 인증 파일 경로를 가져와 절대 경로로 변환
    
    Returns:
        Google Cloud 인증 파일의 절대 경로
    """
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
    
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다.")
    
    # 상대 경로를 절대 경로로 변환
    absolute_path = resolve_credentials_path(credentials_path)
    
    # 파일 존재 확인
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(
            f"Google Cloud 인증 파일을 찾을 수 없습니다: {absolute_path}\n"
            f"원본 경로: {credentials_path}"
        )
    
    return absolute_path