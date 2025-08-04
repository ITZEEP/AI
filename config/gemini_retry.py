import time
import functools
from typing import Any, Callable
import sys
import os

try:
    from google.ai.generativelanguage_v1beta.types import GenerateContentResponse
    from google.api_core.exceptions import InternalServerError
except ImportError:
    try:
        from google.generativeai.types import GenerationConfig
        from google.api_core.exceptions import InternalServerError
    except ImportError:
        try:
            # 일반적인 예외로 대체
            class InternalServerError(Exception):
                """500 Internal Server Error 대체 클래스"""
                pass
        except:
            # 최후의 수단으로 일반 Exception 사용
            InternalServerError = Exception

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.logger_config import get_logger
logger = get_logger(__name__)

def retry_gemini_api(max_retries: int = 5, initial_delay: float = 2.0, backoff_multiplier: float = 1.5, max_delay: float = 30.0):
    """
    Gemini API 호출에 대한 재시도 데코레이터
    
    Args:
        max_retries: 최대 재시도 횟수 (기본 5회)
        initial_delay: 초기 대기시간 (기본 2초)
        backoff_multiplier: 대기시간 증가 배수 (기본 1.5배씩 증가)
        max_delay: 최대 대기시간 제한 (기본 30초)
    
    Usage:
        @retry_gemini_api()  # 기본 설정 사용
        def api_call_method(self):
            return self.llm(messages)
            
        @retry_gemini_api(max_retries=3)  # 커스텀 설정
        def another_method(self):
            return self.llm(messages)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = initial_delay
            
            for attempt in range(max_retries + 1):  # 0부터 시작하므로 +1
                try:
                    # API 호출 시도
                    result = func(*args, **kwargs)
                    
                    # 재시도 후 성공 시 로그
                    if attempt > 0:
                        logger.info(f"성공: {func.__name__} - {attempt + 1}번째 시도에서 성공")
                    
                    return result
                    
                except InternalServerError as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        # 재시도 로그
                        logger.warning(f"재시도 {attempt + 1}/{max_retries}: {func.__name__}")
                        logger.warning(f"에러: {str(e)}")
                        logger.info(f"{current_delay}초 대기 후 재시도...")
                        
                        # 대기
                        time.sleep(current_delay)
                        
                        # 지수적 백오프 (최대 대기시간 제한)
                        current_delay = min(current_delay * backoff_multiplier, max_delay)
                    else:
                        # 최종 실패
                        logger.error(f"최종 실패: {func.__name__} - {max_retries + 1}회 모두 실패")
                        logger.error(f"최종 에러: {str(e)}")
                        
                except Exception as e:
                    # InternalServerError가 아닌 다른 에러는 즉시 발생
                    logger.error(f"즉시 실패: {func.__name__} - {type(e).__name__}: {str(e)}")
                    raise e
            
            # 모든 재시도 실패 시 마지막 예외 발생
            logger.error(f"{func.__name__} 완전 실패")
            raise last_exception
            
        return wrapper
    return decorator