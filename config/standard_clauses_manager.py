"""
config/standard_clauses_manager.py - 간단한 표준 특약 필터링

OCR로 추출된 특약에서 표준 특약들을 제거하고 사용자 정의 특약만 반환
"""

import os
import re
import sys
from typing import List
from difflib import SequenceMatcher

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from config.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    # logger가 없으면 print 사용
    class SimpleLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def debug(self, msg): print(f"DEBUG: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    logger = SimpleLogger()

def filter_custom_clauses_from_list(clauses_list: List[str], similarity_threshold: float = 0.85) -> List[str]:
    """
    특약 리스트에서 사용자 정의 특약만 필터링
    
    Args:
        clauses_list: OCR로 추출된 모든 특약 리스트
        similarity_threshold: 유사도 임계값 (0.85 이상이면 표준 특약으로 판단)
        
    Returns:
        사용자 정의 특약만 포함된 리스트
    """
    if not clauses_list:
        return []
    
    try:
        # 표준 특약 데이터 로드
        from config.standard_clauses import ALL_STANDARD_TERMS, STANDARD_CLAUSES
        
        # 표준 조문도 파싱해서 추가
        standard_clauses = _parse_standard_clauses(STANDARD_CLAUSES)
        all_standards = ALL_STANDARD_TERMS + standard_clauses
        
        logger.info(f"표준 특약 {len(all_standards)}개 로드됨")
        
    except ImportError as e:
        logger.error(f"standard_clauses.py 파일을 찾을 수 없습니다: {e}")
        return clauses_list  # 필터링 없이 원본 반환
    except Exception as e:
        logger.error(f"표준 특약 로드 중 오류: {e}")
        return clauses_list  # 필터링 없이 원본 반환
    
    custom_clauses = []
    removed_count = 0
    
    for clause in clauses_list:
        if not clause or len(clause.strip()) < 10:
            continue
            
        cleaned_clause = _clean_text_for_comparison(clause)
        is_standard = False
        
        # 각 표준 특약과 비교
        for standard in all_standards:
            cleaned_standard = _clean_text_for_comparison(standard)
            similarity = SequenceMatcher(None, cleaned_clause, cleaned_standard).ratio()
            
            if similarity >= similarity_threshold:
                logger.debug(f"표준 특약 제거 (유사도: {similarity:.2f}): {clause[:30]}...")
                is_standard = True
                removed_count += 1
                break
        
        if not is_standard:
            custom_clauses.append(clause)
            logger.debug(f"사용자 정의 특약: {clause[:30]}...")
    
    logger.info(f"필터링 완료: 전체 {len(clauses_list)}개 → 표준 {removed_count}개 제거, 사용자 정의 {len(custom_clauses)}개")
    return custom_clauses


def _parse_standard_clauses(clauses_text: str) -> List[str]:
    """표준 조문 텍스트를 파싱해서 리스트로 변환"""
    if not clauses_text:
        return []
    
    clauses = []
    current_clause = ""
    
    for line in clauses_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # 새로운 조문 시작 (제X조)
        if re.match(r'^제\d+조', line):
            # 이전 조문 저장
            if current_clause:
                clauses.append(_clean_clause_text(current_clause))
            current_clause = line
        else:
            # 기존 조문에 추가
            if current_clause:
                current_clause += " " + line
            else:
                current_clause = line
    
    # 마지막 조문 저장
    if current_clause:
        clauses.append(_clean_clause_text(current_clause))
    
    return clauses


def _clean_clause_text(text: str) -> str:
    """조문 텍스트 정리 (조문 번호 제거)"""
    # 조문 번호와 제목 제거 (예: "제3조(용도변경 및 전대 등)" 제거)
    cleaned = re.sub(r'^제\d+조\([^)]+\)\s*', '', text)
    # 연속 공백 정리
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


def _clean_text_for_comparison(text: str) -> str:
    """텍스트 정리 (비교용)"""
    # 불릿 포인트 제거
    cleaned = re.sub(r'^[Ÿ•○●▪▫\-]\s*', '', text)
    # 조문 번호 제거
    cleaned = re.sub(r'^제\d+조\([^)]+\)\s*', '', cleaned)
    # 체크박스와 빈칸 제거
    cleaned = re.sub(r'[□■▢▣]', '', cleaned)
    cleaned = re.sub(r'_+', '', cleaned)
    # 괄호 안 내용 제거 (선택 사항들)
    cleaned = re.sub(r'\([^)]*\)', '', cleaned)
    # 연속 공백 정리
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # 특수문자 정리
    cleaned = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩※＊*]', '', cleaned)
    
    return cleaned.strip()