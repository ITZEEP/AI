import pdfplumber
import re
import sys
import json
import os
import fitz
import cv2
import numpy as np
from google.cloud import vision
from dotenv import load_dotenv
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

# ✅ 환경 변수 및 Vision API 클라이언트 설정
_vision_client = None
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.logger_config import get_logger
logger = get_logger(__name__)

def get_vision_client():
    global _vision_client
    if _vision_client is None:
        load_dotenv()
        json_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not json_path or not os.path.exists(json_path):
            raise RuntimeError(
                "환경 변수 GOOGLE_APPLICATION_CREDENTIALS가 없거나 경로가 잘못되었습니다.")
        _vision_client = vision.ImageAnnotatorClient()
    return _vision_client


def pixmap_to_bgr(pix):
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def ocr_google_vision(image_np):
    success, encoded_image = cv2.imencode('.png', image_np)
    if not success:
        raise RuntimeError("이미지 인코딩 실패")
    image = vision.Image(content=encoded_image.tobytes())
    response = get_vision_client().document_text_detection(image=image)
    if response.error.message:
        raise RuntimeError(f"Google Vision API 오류: {response.error.message}")
    return response


def is_image_based_pdf(file_path):
    """PDF가 이미지 기반인지 확인"""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:3]:  # 처음 3페이지만 확인
                text = page.extract_text()
                if text and len(text.strip()) > 100:  # 충분한 텍스트가 있으면 텍스트 기반
                    return False
        return True  # 텍스트가 거의 없으면 이미지 기반
    except Exception as e:
        print(f"PDF 유형 확인 중 오류 발생: {e}")
        return True


def extract_text_from_image_pdf(file_path):
    """이미지 기반 PDF에서 OCR로 텍스트 추출"""
    all_text = ""
    with fitz.open(file_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            image = pixmap_to_bgr(pix)

            try:
                response = ocr_google_vision(image)
                if response.full_text_annotation:
                    page_text = response.full_text_annotation.text
                    all_text += page_text + "\n"
                    print(f"=== 페이지 {page_num + 1} OCR 결과 ===")
                    print(page_text)
                    print("=" * 50)
            except Exception as e:
                print(f"페이지 {page_num + 1} OCR 처리 중 오류: {e}")
                continue

    return all_text


def debug_ocr_line_by_line(ocr_text):
    """OCR 텍스트를 라인별로 분석"""
    lines = ocr_text.split('\n')
    print(f"\n=== OCR 라인별 분석 (총 {len(lines)}줄) ===")
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if line:  # 빈 줄 제외
            print(f"{i:3d}: {line}")
    
    print("=" * 50)
    
    # 특정 키워드 검색
    keywords = ["갑구", "을구", "소유자", "채권최고액", "근저당", "채무자", "발행일"]
    print("\n=== 키워드 검색 결과 ===")
    
    for keyword in keywords:
        found_lines = []
        for i, line in enumerate(lines, 1):
            if keyword in line:
                found_lines.append(f"  {i:3d}: {line.strip()}")
        
        if found_lines:
            print(f"\n'{keyword}' 포함 라인:")
            for found_line in found_lines:
                print(found_line)
        else:
            print(f"\n'{keyword}': 발견되지 않음")
    
    print("=" * 50)


def parse_ocr_text_for_registration(ocr_text):
    """OCR로 추출된 텍스트를 파싱하여 등기부 정보 추출 - 텍스트 기반과 동일한 구조로"""
    result = {
        "표제부": {
            "소재지번_건물명칭": None,
            "건물번호": None,
            "건물내역": None
        },
        "발행일": None,
        "갑구": [],
        "을구": [],
        "법적상태": {
            "가압류_여부": False,
            "경매_여부": False,
            "소송_여부": False,
            "압류_여부": False
        }
    }

    lines = ocr_text.split('\n')
    
    print(f"OCR 추출된 라인 수: {len(lines)}")

    # 발행일 추출
    for line in lines:
        match = re.search(r"발행일\s*(\d{4}[./]\d{2}[./]\d{2})", line)
        if match:
            result["발행일"] = match.group(1).replace('.', '/').replace('-', '/')
            print(f"발행일 발견: {result['발행일']}")
            break

    # 소재지번 추출 (도로명주소 포함) - 더 유연한 패턴으로
    for line in lines:
        # 주소 키워드와 지역명이 포함된 라인 찾기
        if any(addr_keyword in line for addr_keyword in [
            "서울", "경기", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
            "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
        ]):
            # 소재지번, 건물명칭, 도로명주소 등의 키워드가 없어도 주소로 인식
            if len(line.strip()) > 10:  # 충분히 긴 주소
                result["표제부"]["소재지번_건물명칭"] = line.strip()
                print(f"주소 발견: {line.strip()}")
                break
        
        # 기존 키워드 기반 검색도 유지
        if (any(keyword in line for keyword in ["소재지번", "건물명칭", "도로명주소"]) and
                any(addr_keyword in line for addr_keyword in [
                    "서울", "경기", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
                    "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
                ])):
            result["표제부"]["소재지번_건물명칭"] = line.strip()
            print(f"키워드 기반 주소 발견: {line.strip()}")
            break

    # 건물번호 추출
    for line in lines:
        if re.search(r'제?\d+층.*제?\d+호|제?\d+호', line) and "m2" not in line and "㎡" not in line:
            result["표제부"]["건물번호"] = line.strip()
            print(f"건물번호 발견: {line.strip()}")
            break

    # 건물내역 추출
    for line in lines:
        if re.search(r'.*구\s*조.*\d+\.?\d*\s*(m2|㎡)', line):
            result["표제부"]["건물내역"] = line.strip()
            print(f"건물내역 발견: {line.strip()}")
            break

    # 갑구, 을구 섹션 파싱 - 구조화된 데이터로 변환
    current_section = None
    gap_gu_data = []
    eul_gu_data = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "갑구" in line or "갑 구" in line:
            current_section = "갑구"
            print(f"갑구 섹션 시작")
            continue
        elif "을구" in line or "을 구" in line:
            current_section = "을구"
            print(f"을구 섹션 시작")
            continue

        if current_section == "갑구":
            _parse_gabgu_line_from_ocr(line, gap_gu_data)
        elif current_section == "을구":
            _parse_eulgu_line_from_ocr(line, eul_gu_data)

    result["갑구"] = gap_gu_data if gap_gu_data else ["데이터 없음"]
    result["을구"] = eul_gu_data if eul_gu_data else ["데이터 없음"]
    
    print(f"갑구 데이터: {len(gap_gu_data)}개")
    print(f"을구 데이터: {len(eul_gu_data)}개")

    return result


def _parse_gabgu_line_from_ocr(line, gap_gu_data):
    """갑구 OCR 라인 파싱 - 소유자 정보 추출"""
    if "순위번호" in line or "등기목적" in line:
        return
    
    # 소유자 정보가 포함된 라인 찾기 - 더 유연한 패턴
    if "소유자" in line or (re.search(r'\d{6}-[\d*]{7}', line) and any(keyword in line for keyword in ["홍길동", "김", "이", "박", "최", "정", "강", "조", "윤", "장"])):
        row_data = []
        
        # 순위번호 추출 - 라인 시작 부분의 숫자
        rank_match = re.search(r'^[\s]*(\d+)', line)
        if rank_match:
            rank_number = int(rank_match.group(1))
            row_data.append({"순위번호": rank_number})
            print(f"갑구 순위번호 발견: {rank_number}")
        
        # 소유자명 추출 - 여러 패턴 시도
        owner_patterns = [
            r'소유자\s+([가-힣]{2,4})',  # 소유자 홍길동
            r'([가-힣]{2,4})\s+\d{6}-[\d*]{7}',  # 홍길동 123456-1234567
            r'소유권이전등기\s+([가-힣]{2,4})',  # 소유권이전등기 홍길동
        ]
        
        owner_name = None
        for pattern in owner_patterns:
            owner_match = re.search(pattern, line)
            if owner_match:
                owner_name = owner_match.group(1)
                row_data.append({"소유자명": owner_name})
                print(f"갑구 소유자명 발견: {owner_name}")
                break
        
        # 주민번호 추출
        id_match = re.search(r'(\d{6}-[\d*]{7})', line)
        if id_match:
            resident_id = id_match.group(1)
            row_data.append({"주민번호": resident_id})
            print(f"갑구 주민번호 발견: {resident_id}")
        
        # 거래가액 추출
        price_match = re.search(r'거래가액\s*(금[\d,]+원)', line)
        if price_match:
            price = price_match.group(1)
            row_data.append({"거래가액": price})
            print(f"갑구 거래가액 발견: {price}")
        
        # 권리자 및 기타사항으로 전체 라인 추가
        row_data.append({"권리자 및 기타사항": line})
        
        if row_data:
            gap_gu_data.append(row_data)


def _parse_eulgu_line_from_ocr(line, eul_gu_data):
    """을구 OCR 라인 파싱 - 근저당권 정보 추출"""
    if "순위번호" in line or "등기목적" in line:
        return
    
    # 근저당권 정보가 포함된 라인 찾기 - 더 유연한 패턴
    if any(keyword in line for keyword in ["채권최고액", "근저당권", "채무자", "금", "원"]) and any(keyword in line for keyword in ["은행", "조합", "회사", "기관"]):
        row_data = []
        
        # 순위번호 추출 - 라인 시작 부분의 숫자
        rank_match = re.search(r'^[\s]*(\d+)', line)
        if rank_match:
            rank_number = int(rank_match.group(1))
            row_data.append({"순위번호": rank_number})
            print(f"을구 순위번호 발견: {rank_number}")
        
        # 채권최고액 추출 - 여러 패턴 시도
        amount_patterns = [
            r'채권최고액\s*(금[\d,]+원)',
            r'(금[\d,]+원)',
            r'금액\s*([\d,]+)원?',
            r'([\d,]+)원'
        ]
        
        for pattern in amount_patterns:
            amount_match = re.search(pattern, line)
            if amount_match:
                amount = amount_match.group(1)
                if "금" not in amount:
                    amount = f"금{amount}원"
                row_data.append({"채권최고액": amount})
                print(f"을구 채권최고액 발견: {amount}")
                break
        
        # 채무자 추출 - 사람 이름 패턴
        debtor_patterns = [
            r'채무자\s+([가-힣]{2,4})',
            r'([가-힣]{2,4})\s+\d{6}-[\d*]{7}',  # 이름 + 주민번호
        ]
        
        for pattern in debtor_patterns:
            debtor_match = re.search(pattern, line)
            if debtor_match:
                debtor = debtor_match.group(1)
                row_data.append({"채무자": debtor})
                print(f"을구 채무자 발견: {debtor}")
                break
        
        # 근저당권자 추출 - 기관명 패턴
        mortgagee_patterns = [
            r'근저당권자\s+([^0-9\n]+?)(?:\s+\d|$)',
            r'(주식회사[가-힣]+)',
            r'([가-힣]+은행)',
            r'([가-힣]+조합)',
        ]
        
        for pattern in mortgagee_patterns:
            mortgagee_match = re.search(pattern, line)
            if mortgagee_match:
                mortgagee = mortgagee_match.group(1).strip()
                if len(mortgagee) > 1:  # 최소 2글자 이상
                    row_data.append({"근저당권자": mortgagee})
                    print(f"을구 근저당권자 발견: {mortgagee}")
                    break
        
        # 권리자 및 기타사항으로 전체 라인 추가
        row_data.append({"권리자 및 기타사항": line})
        
        if row_data:
            eul_gu_data.append(row_data)


def extract_all_real_estate_info(file_path):
    """등기부등본 전체 정보 추출 - 최신 소유자 및 다중 근저당권 지원"""
    result = {
        "file_name": os.path.basename(file_path),
        "extracted_at": datetime.now().isoformat(),
        "표제부": {
            "소재지번_건물명칭": None,
            "건물번호": None,
            "건물내역": None
        },
        "발행일": None,
        "갑구": [],
        "을구": [],
        "법적상태": {
            "가압류_여부": False,
            "경매_여부": False,
            "소송_여부": False,
            "압류_여부": False
        }
    }

    # PDF 유형 확인
    if is_image_based_pdf(file_path):
        print("이미지 기반 PDF로 감지됨. OCR 처리를 시작합니다...")
        all_text_content = extract_text_from_image_pdf(file_path)

        if all_text_content.strip():
            # OCR 결과 디버깅
            debug_ocr_line_by_line(all_text_content)
            
            # OCR 텍스트 파싱 (텍스트 기반과 동일한 구조로)
            ocr_result = parse_ocr_text_for_registration(all_text_content)

            # 결과 병합
            for key in ["표제부", "발행일", "갑구", "을구", "법적상태"]:
                if ocr_result.get(key):
                    result[key] = ocr_result[key]
            
            # ✅ 이미지 기반에서도 소유자 및 근저당권 정보 추출
            print(f"\n=== 소유자/근저당권 정보 추출 시작 ===")
            latest_owner_info = _extract_latest_owner_info(result["갑구"])
            result["소유자정보"] = latest_owner_info
            print(f"추출된 소유자 정보: {latest_owner_info}")
            
            mortgage_info = _extract_all_mortgage_info(result["을구"])
            result["근저당권정보"] = mortgage_info
            print(f"추출된 근저당권 정보: {mortgage_info}")
            
            # ✅ risk_analysis_ready 섹션 생성
            result["risk_analysis_ready"] = _prepare_risk_analysis_data(result, latest_owner_info, mortgage_info)
            
        else:
            print("OCR에서 텍스트를 추출하지 못했습니다.")
            all_text_content = ""
    else:
        print("텍스트 기반 PDF로 감지됨. 일반 추출을 진행합니다...")
        # 기존 pdfplumber 기반 처리
        result = extract_text_based_pdf(file_path, result)
        all_text_content = extract_all_text_from_pdf(file_path)

    # 법적 상태 확인 (이미지/텍스트 공통)
    check_legal_status(all_text_content, result["법적상태"])

    return result


def extract_text_based_pdf(file_path, result):
    """텍스트 기반 PDF 처리 (수정된 로직)"""
    gap_gu_data, eul_gu_data = [], []
    gap_gu_raw_texts, eul_gu_raw_texts = [], []
    title_text_blocks = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                title_text_blocks.append(page_text)

                if result["발행일"] is None:
                    match = re.search(r"발행일\s*(\d{4}/\d{2}/\d{2})", page_text)
                    if match:
                        result["발행일"] = match.group(1).strip()

            tables = page.find_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "join_y_tolerance": 40
                }
            )

            current_section = None
            header_row = None

            for table in tables:
                table_data = table.extract()
                if not table_data:
                    continue

                # 표제부 정보 추출
                extract_title_section_info(table_data, result["표제부"])

                for row in table_data:
                    if not row:
                        continue

                    row_text = " ".join([str(cell)
                                        for cell in row if cell]).strip()

                    if "갑구" in row_text or "갑 구" in row_text:
                        current_section = "갑구"
                        header_row = None
                        continue
                    elif "을구" in row_text or "을 구" in row_text:
                        current_section = "을구"
                        header_row = None
                        continue

                    if current_section and any("기록사항 없음" in str(cell) for cell in row):
                        text = " ".join([str(cell)
                                        for cell in row if cell]).strip()
                        if current_section == "갑구":
                            gap_gu_raw_texts.append(text)
                        elif current_section == "을구":
                            eul_gu_raw_texts.append(text)
                        continue

                    if current_section and ("순위번호" in row_text or "등기목적" in row_text):
                        header_row = row
                        continue

                    if current_section and header_row:
                        row_data = []
                        
                        # 순위번호 추출
                        rank_number = None
                        for j, cell in enumerate(row):
                            if cell and str(cell).strip():
                                clean_cell = str(cell).replace("\n", " ").strip()
                                header = header_row[j] if j < len(header_row) else f"컬럼{j}"
                                
                                # 순위번호 찾기
                                if str(header).strip() == "순위번호" or j == 0:
                                    rank_match = re.search(r'(\d+)', clean_cell)
                                    if rank_match:
                                        rank_number = int(rank_match.group(1))
                                
                                row_data.append({str(header).strip(): clean_cell})
                                
                                # 권리자 정보 상세 파싱
                                if str(header).strip() == "권리자 및 기타사항":
                                    text = clean_cell
                                    if current_section == "갑구":
                                        owner = re.search(r"소유자\s+([^\s]+)", text)
                                        price = re.search(r"거래가액\s*(금[\d,]+원)", text)
                                        # 주민번호 추출 (6자리-7자리 또는 마스킹된 형태)
                                        resident_id = re.search(r'(\d{6}-[\d*]{7})', text)
                                        
                                        if owner:
                                            row_data.append({"소유자명": owner.group(1)})
                                            row_data.append({"순위번호": rank_number or 0})
                                        if resident_id:
                                            row_data.append({"주민번호": resident_id.group(1)})
                                        if price:
                                            row_data.append({"거래가액": price.group(1)})
                                            
                                    elif current_section == "을구":
                                        max_amt = re.search(r"채권최고액\s*(금[\d,]+원)", text)
                                        debtor = re.search(r"채무자\s+([^\s]+)", text)
                                        mortgagee = re.search(r"근저당권자\s+([^\d\n]+)", text)
                                        
                                        if max_amt:
                                            row_data.append({"채권최고액": max_amt.group(1)})
                                            row_data.append({"순위번호": rank_number or 0})
                                        if debtor:
                                            row_data.append({"채무자": debtor.group(1).strip()})
                                        if mortgagee:
                                            row_data.append({"근저당권자": mortgagee.group(1).strip()})

                        if row_data:
                            if current_section == "갑구":
                                gap_gu_data.append(row_data)
                            elif current_section == "을구":
                                eul_gu_data.append(row_data)

    result["갑구"] = gap_gu_data if gap_gu_data else gap_gu_raw_texts or ["데이터 없음"]
    result["을구"] = eul_gu_data if eul_gu_data else eul_gu_raw_texts or ["데이터 없음"]
    
    # ✅ 최신 소유자 정보 추출 및 추가
    latest_owner_info = _extract_latest_owner_info(result["갑구"])
    result["소유자정보"] = latest_owner_info
    
    # ✅ 다중 근저당권 정보 추출 및 추가  
    mortgage_info = _extract_all_mortgage_info(result["을구"])
    result["근저당권정보"] = mortgage_info
    
    # ✅ risk_analysis_ready 섹션 업데이트
    result["risk_analysis_ready"] = _prepare_risk_analysis_data(result, latest_owner_info, mortgage_info)

    return result


def extract_all_text_from_pdf(file_path):
    """PDF에서 모든 텍스트 추출"""
    all_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"
    return all_text


def check_legal_status(text_content, legal_status):
    """전체 텍스트에서 법적 상태 관련 키워드 확인"""

    # 가압류 관련 키워드
    seizure_keywords = ["가압류", "가압류등기", "가압류신청", "가압류명령", "가압류결정"]
    seizure_negative = ["가압류해제", "가압류취소", "가압류말소"]

    # 경매 관련 키워드
    auction_keywords = ["경매", "경매개시", "경매신청", "경매개시결정", "강제경매", "임의경매"]
    auction_negative = ["경매취소", "경매말소"]

    # 소송 관련 키워드
    lawsuit_keywords = ["소송", "민사소송", "소유권이전등기청구", "소유권이전등기말소", "소유권확인", "손해배상"]
    lawsuit_negative = ["소송취소", "소송말소"]

    # 압류 관련 키워드
    attachment_keywords = ["압류", "압류등기", "압류신청", "압류명령", "압류결정", "강제집행"]
    attachment_negative = ["압류취소", "압류말소"]

    # 키워드 검사
    for keyword in seizure_keywords:
        if keyword in text_content:
            if not any(neg in text_content for neg in seizure_negative):
                legal_status["가압류_여부"] = True
                logger.info(f"가압류 키워드 발견: {keyword}")
                break

    for keyword in auction_keywords:
        if keyword in text_content:
            if not any(neg in text_content for neg in auction_negative):
                legal_status["경매_여부"] = True
                logger.info(f"경매 키워드 발견: {keyword}")
                break

    for keyword in lawsuit_keywords:
        if keyword in text_content:
            if not any(neg in text_content for neg in lawsuit_negative):
                legal_status["소송_여부"] = True
                logger.info(f"소송 키워드 발견: {keyword}")
                break

    for keyword in attachment_keywords:
        if keyword in text_content:
            if not any(neg in text_content for neg in attachment_negative):
                legal_status["압류_여부"] = True
                logger.info(f"압류 키워드 발견: {keyword}")
                break


def extract_title_section_info(table_data, title_data):
    """표제부 정보를 추출하여 title_data 딕셔너리에 저장"""
    _find_location_info(table_data, title_data)
    _find_jeonyu_info(table_data, title_data)


def _find_location_info(table_data, extracted_data):
    """소재지번_건물명칭 정보 추출"""
    if extracted_data["소재지번_건물명칭"]:
        return

    for row in table_data:
        if not row:
            continue
        for cell in row:
            if not cell:
                continue

            cell_text = str(cell).strip()
            if any(keyword in cell_text for keyword in ["소재지번", "건물명칭", "도로명주소"]):
                for row_cell in row:
                    if row_cell and str(row_cell).strip():
                        cell_content = str(row_cell).strip()
                        if any(addr_keyword in cell_content for addr_keyword in [
                            "서울", "경기", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
                            "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
                        ]):
                            extracted_data["소재지번_건물명칭"] = cell_content.replace(
                                "\n", " ").replace("  ", " ").strip()
                            return


def _find_jeonyu_info(table_data, extracted_data):
    """전유부분의 건물번호와 건물내역 정보 추출"""
    if extracted_data["건물번호"] and extracted_data["건물내역"]:
        return

    jeonyu_section_start = -1
    for row_idx, row in enumerate(table_data):
        if not row:
            continue
        row_text = " ".join([str(cell) for cell in row if cell])
        if "전유부분" in row_text:
            jeonyu_section_start = row_idx
            break

    if jeonyu_section_start == -1:
        return

    for analysis_row_idx in range(jeonyu_section_start + 1, len(table_data)):
        current_row = table_data[analysis_row_idx]
        if not current_row:
            continue

        building_info = _analyze_row_for_building_info(current_row)

        if building_info["building_number"] and not extracted_data["건물번호"]:
            extracted_data["건물번호"] = building_info["building_number"].replace(
                "\n", " ").replace("  ", " ").strip()

        if building_info["building_detail"] and not extracted_data["건물내역"]:
            extracted_data["건물내역"] = building_info["building_detail"].replace(
                "\n", " ").replace("  ", " ").strip()

        if extracted_data["건물번호"] and extracted_data["건물내역"]:
            break


def _analyze_row_for_building_info(row):
    """행을 분석하여 건물번호와 건물내역 정보 추출"""
    building_info = {
        "building_number": None,
        "building_detail": None
    }

    if not row:
        return building_info

    for cell_idx, cell in enumerate(row):
        if not cell:
            continue

        cell_text = str(cell).strip()
        if not cell_text:
            continue

        # 건물번호 패턴 검사
        building_number_patterns = [
            r'제\d+층.*제\d+호',
            r'\d+층.*\d+호',
            r'제\d+호',
            r'\d+호',
        ]
        for pattern in building_number_patterns:
            if re.search(pattern, cell_text) and not building_info["building_number"]:
                if "m2" not in cell_text and "㎡" not in cell_text:
                    building_info["building_number"] = cell_text
                    break

        # 건물내역 패턴 검사
        building_detail_patterns = [
            r'.*구\s*조.*\d+\.?\d*\s*m2',
            r'.*구\s*조.*\d+\.?\d*\s*㎡',
            r'.*구조.*\d+\.?\d*\s*m2',
            r'.*구조.*\d+\.?\d*\s*㎡',
        ]
        for pattern in building_detail_patterns:
            if re.search(pattern, cell_text) and not building_info["building_detail"]:
                building_info["building_detail"] = cell_text
                break

        # 면적 정보가 있는 경우 구조 정보와 결합
        area_pattern = r'\d+\.?\d*\s*(m2|㎡)'
        if re.search(area_pattern, cell_text) and not building_info["building_detail"]:
            structure_info = _find_structure_in_same_row(row, cell_idx)
            if structure_info:
                building_info["building_detail"] = f"{structure_info} {cell_text}"

    return building_info


def _find_structure_in_same_row(row, exclude_cell_idx):
    """같은 행에서 구조 정보 찾기"""
    structure_keywords = ["콘크리트", "구조", "철골", "철근", "목구조", "벽돌", "블록"]
    for cell_idx, cell in enumerate(row):
        if cell_idx == exclude_cell_idx or not cell:
            continue

        cell_text = str(cell).strip()
        if any(keyword in cell_text for keyword in structure_keywords):
            if not re.search(r'\d+\.?\d*\s*(m2|㎡)', cell_text):
                return cell_text
    return None


def _extract_latest_owner_info(gabgu_data) -> Dict[str, Any]:
    """갑구 데이터에서 최신 소유자 정보 추출"""
    latest_owner_info = {
        "소유자명": None,
        "주민번호": None,
        "소유권이전일": None,
        "거래가액": None
    }
    
    if not gabgu_data:
        return latest_owner_info
    
    latest_rank = 0
    
    try:
        for item in gabgu_data:
            if isinstance(item, list):
                current_rank = 0
                current_owner = None
                current_id = None
                current_date = None
                current_price = None
                
                for detail in item:
                    if isinstance(detail, dict):
                        # 순위번호 찾기
                        if "순위번호" in detail:
                            rank_str = str(detail["순위번호"])
                            rank_match = re.search(r'(\d+)', rank_str)
                            if rank_match:
                                current_rank = int(rank_match.group(1))
                                
                        # 소유자명 찾기
                        if "소유자명" in detail:
                            current_owner = detail["소유자명"]
                            
                        # 주민번호 찾기
                        if "주민번호" in detail:
                            current_id = detail["주민번호"]
                            
                        # 권리자 및 기타사항에서 주민번호 추출 (위에서 못 찾은 경우)
                        if not current_id and "권리자 및 기타사항" in detail:
                            text = detail["권리자 및 기타사항"]
                            # 주민번호 패턴 (6자리-7자리 또는 마스킹된 형태)
                            id_match = re.search(r'(\d{6}-[\d*]{7})', text)
                            if id_match:
                                current_id = id_match.group(1)
                                
                        # 등기원인에서 날짜 추출
                        if "등 기 원 인" in detail:
                            date_text = detail["등 기 원 인"]
                            date_match = re.search(r'(\d{4}년\d{1,2}월\d{1,2}일)', date_text)
                            if date_match:
                                current_date = date_match.group(1)
                                
                        # 거래가액
                        if "거래가액" in detail:
                            current_price = detail["거래가액"]
                
                # 순위번호가 더 높으면 최신 정보로 업데이트
                if current_rank >= latest_rank and current_owner:
                    latest_rank = current_rank
                    latest_owner_info["소유자명"] = current_owner
                    latest_owner_info["주민번호"] = current_id
                    latest_owner_info["소유권이전일"] = current_date
                    latest_owner_info["거래가액"] = current_price
                    
    except Exception as e:
        logger.error(f"최신 소유자 정보 추출 실패: {e}")
    
    logger.info(f"✅ 최신 소유자 정보: {latest_owner_info['소유자명']} (순위: {latest_rank})")
    return latest_owner_info


def _extract_all_mortgage_info(eulgu_data) -> Dict[str, Any]:
    """을구 데이터에서 모든 근저당권 정보 추출"""
    mortgage_info = {
        "근저당권목록": [],
        "총근저당권액": 0,
        "근저당권개수": 0
    }
    
    if not eulgu_data:
        return mortgage_info
    
    try:
        for item in eulgu_data:
            if isinstance(item, list):
                current_mortgage = {}
                
                for detail in item:
                    if isinstance(detail, dict):
                        # 순위번호
                        if "순위번호" in detail:
                            current_mortgage["순위번호"] = detail["순위번호"]
                            
                        # 채권최고액
                        if "채권최고액" in detail:
                            amount_str = detail["채권최고액"]
                            amount = _extract_amount_from_korean(amount_str)
                            current_mortgage["채권최고액"] = amount
                            current_mortgage["채권최고액_원문"] = amount_str
                            if amount:
                                mortgage_info["총근저당권액"] += amount
                                
                        # 채무자
                        if "채무자" in detail:
                            current_mortgage["채무자"] = detail["채무자"]
                            
                        # 근저당권자
                        if "근저당권자" in detail:
                            current_mortgage["근저당권자"] = detail["근저당권자"]
                            
                        # 등기원인
                        if "등 기 원 인" in detail:
                            current_mortgage["등기원인"] = detail["등 기 원 인"]
                
                # 근저당권 정보가 있으면 목록에 추가
                if current_mortgage and current_mortgage.get("채권최고액"):
                    mortgage_info["근저당권목록"].append(current_mortgage)
                    mortgage_info["근저당권개수"] += 1
                    
    except Exception as e:
        logger.error(f"근저당권 정보 추출 실패: {e}")
    
    logger.info(f"✅ 근저당권 {mortgage_info['근저당권개수']}개 추출, 총액: {mortgage_info['총근저당권액']:,}원")
    return mortgage_info


def _prepare_risk_analysis_data(result, owner_info, mortgage_info) -> Dict[str, Any]:
    """위험도 분석을 위한 데이터 준비 - 요구사항에 맞는 형식으로 출력"""
    
    # region_address에서 도로명주소 분리 - None 체크 추가
    full_address = result["표제부"].get("소재지번_건물명칭") or ""
    region_address = full_address
    road_address = ""
    
    # [도로명주소] 이후 부분 추출 및 region_address에서 제거
    if full_address and "[도로명주소]" in full_address:
        parts = full_address.split("[도로명주소]")
        if len(parts) > 1:
            region_address = parts[0].strip()  # 도로명주소 앞부분만
            road_address = parts[1].strip()    # 도로명주소 부분만
    
    risk_data = {
        "region_address": region_address,
        "road_address": road_address,
        "owner_name": owner_info.get("소유자명") or "",
        "owner_birth_date": _parse_birth_date_from_id(owner_info.get("주민번호")),
        "has_seizure": result["법적상태"]["가압류_여부"],
        "has_auction": result["법적상태"]["경매_여부"], 
        "has_litigation": result["법적상태"]["소송_여부"],
        "has_attachment": result["법적상태"]["압류_여부"]
    }
    
    # 근저당권 정보 추가 - 새로운 형식으로
    if mortgage_info["근저당권목록"]:
        # 가장 큰 근저당권에서 채무자 정보 추출 (최상위 debtor용)
        max_mortgage = max(mortgage_info["근저당권목록"], 
                          key=lambda x: x.get("채권최고액", 0))
        risk_data["debtor"] = max_mortgage.get("채무자") or ""
        
        # 모든 근저당권 정보를 새로운 형식으로 변환
        mortgagee_list = []
        for mortgage in mortgage_info["근저당권목록"]:
            mortgage_item = {
                "priorityNumber": mortgage.get("순위번호") or 0,
                "MaxClaimAmount": mortgage.get("채권최고액") or 0,
                "debtor": mortgage.get("채무자") or "",
                "mortgagee": mortgage.get("근저당권자") or ""
            }
            mortgagee_list.append(mortgage_item)
        
        risk_data["mortgageeList"] = mortgagee_list
    else:
        risk_data["debtor"] = ""
        risk_data["mortgageeList"] = []
    
    return risk_data


def _parse_birth_date_from_id(resident_id: str) -> Optional[str]:
    """주민번호에서 생년월일 추출 (YYYY-MM-DD 형식)"""
    if not resident_id:
        return None
        
    try:
        # 6자리-7자리 형태에서 앞 6자리 추출
        id_match = re.match(r'(\d{6})-', resident_id)
        if id_match:
            birth_part = id_match.group(1)
            year_part = birth_part[:2]
            month_part = birth_part[2:4]
            day_part = birth_part[4:6]
            
            # 연도 보정 (00-99 → 1900-1999 또는 2000-2099)
            year = int(year_part)
            if year >= 0 and year <= 21:  # 2000-2021년생
                full_year = 2000 + year
            else:  # 1922-1999년생
                full_year = 1900 + year
                
            return f"{full_year}-{month_part}-{day_part}"
            
    except Exception as e:
        logger.error(f"생년월일 파싱 실패: {e}")
        
    return None


def _extract_amount_from_korean(amount_str: str) -> Optional[int]:
    """한국어 금액 표기에서 숫자 추출 (예: '금1,000,000원' → 1000000)"""
    if not amount_str:
        return None
    
    try:
        numbers = re.findall(r'[\d,]+', amount_str)
        if numbers:
            return int(numbers[0].replace(',', ''))
    except Exception as e:
        logger.error(f"금액 추출 실패: {e}")
    
    return None


def extract_title_info(lines):
    """기존 함수 - 더 이상 사용하지 않지만 하위 호환성을 위해 유지"""
    title_info = {}
    for line in lines:
        if "표제부" in line:
            title_info["표제부 라벨"] = line
    return title_info


def save_json(output_dict, output_path):
    """JSON 파일로 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)


# ✅ 테스트용 함수 추가
def test_risk_analysis_format():
    """risk_analysis_ready 형식 테스트"""
    sample_data = {
        "risk_analysis_ready": {
            "region_address": "서울특별시 송파구 신천동 29 롯데월드타워앤드롯데월드몰 제월드타워동",
            "road_address": "서울특별시 송파구 올림픽로 300",
            "owner_name": "한은숙",
            "owner_birth_date": "1956-02-17",
            "has_seizure": False,
            "has_auction": False,
            "has_litigation": False,
            "has_attachment": False,
            "debtor": "한은숙",
            "mortgageeList": [
                {
                    "priorityNumber": 1,
                    "MaxClaimAmount": 4680000000,
                    "debtor": "한은숙",
                    "mortgagee": "주식회사하나은행"
                },
                {
                    "priorityNumber": 2,
                    "MaxClaimAmount": 1728000000,
                    "debtor": "한은숙",
                    "mortgagee": "영등포농업협동조합"
                }
            ]
        }
    }
    
    print("=== 기대하는 출력 형식 ===")
    print(json.dumps(sample_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="등기부등본 PDF 통합 정보 추출기")
    parser.add_argument("file", type=str, nargs='?', help="PDF 파일 경로")
    parser.add_argument("--test", action="store_true", help="테스트 형식 출력")
    parser.add_argument("--debug", action="store_true", help="OCR 디버깅 모드")
    args = parser.parse_args()

    if args.test:
        test_risk_analysis_format()
        exit(0)

    if not args.file:
        print("PDF 파일 경로를 입력해주세요.")
        print("사용법: python register_parser.py <파일경로>")
        print("테스트: python register_parser.py --test")
        print("디버깅: python register_parser.py <파일경로> --debug")
        exit(1)

    data = extract_all_real_estate_info(args.file)
    
    # OCR 디버깅 모드인 경우 전체 결과도 출력
    if args.debug:
        print("\n=== 전체 추출 결과 ===")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print("=" * 50)
    
    # risk_analysis_ready 섹션만 저장
    if "risk_analysis_ready" in data:
        risk_only = {"risk_analysis_ready": data["risk_analysis_ready"]}
        
        output_dir = "C:/LLM/data/output/registration_json"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        output_path = os.path.join(output_dir, f"{base_name}_등기부통합.json")
        save_json(risk_only, output_path)
        
        # 콘솔에도 출력
        print("\n=== Risk Analysis Ready 출력 ===")
        print(json.dumps(risk_only, ensure_ascii=False, indent=2))
        print(f"\n✅ Risk Analysis Ready 저장: {output_path}")
    else:
        print("❌ risk_analysis_ready 섹션을 생성할 수 없습니다.")