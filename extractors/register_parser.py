import pdfplumber
import re
import json
import os
import sys
import fitz
import cv2
import numpy as np
from google.cloud import vision
from dotenv import load_dotenv
import logging
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from config.path_resolver import get_google_credentials_path
except ImportError:
    print(f"현재 디렉토리: {current_dir}")
    print(f"프로젝트 루트: {project_root}")
    print(f"sys.path: {sys.path}")
    raise

# OK 환경 변수 및 Vision API 클라이언트 설정
_vision_client = None


def get_vision_client():
    global _vision_client
    if _vision_client is None:
        load_dotenv()
        
        # 상대 경로를 절대 경로로 변환
        try:
            json_path = get_google_credentials_path()
            # 환경 변수를 절대 경로로 업데이트
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path
        except (ValueError, FileNotFoundError) as e:
            raise RuntimeError(f"Google Cloud 인증 설정 오류: {e}")
        
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
            except Exception as e:
                print(f"페이지 {page_num + 1} OCR 처리 중 오류: {e}")
                continue

    return all_text


def parse_ocr_text_for_registration(ocr_text):
    """OCR로 추출된 텍스트를 파싱하여 등기부 정보 추출"""
    result = {
        "표제부": {
            "소재지번_건물명칭": None,
            "건물번호": None,
            "건물내역": None
        },
        "발행일": None,
        "갑구": [],
        "을구": []
    }

    lines = ocr_text.split('\n')

    # 발행일 추출
    for line in lines:
        match = re.search(r"발행일\s*(\d{4}[./]\d{2}[./]\d{2})", line)
        if match:
            result["발행일"] = match.group(1).replace('.', '/').replace('-', '/')
            break

    # 소재지번 추출
    for line in lines:
        if (any(keyword in line for keyword in ["소재지번", "건물명칭", "도로명주소"]) and
                any(addr_keyword in line for addr_keyword in [
                    "서울", "경기", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
                    "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
                ])):
            result["표제부"]["소재지번_건물명칭"] = line.strip()
            break

    # 건물번호 추출
    for line in lines:
        if re.search(r'제?\d+층.*제?\d+호|제?\d+호', line) and "m2" not in line and "㎡" not in line:
            result["표제부"]["건물번호"] = line.strip()
            break

    # 건물내역 추출
    for line in lines:
        if re.search(r'.*구\s*조.*\d+\.?\d*\s*(m2|㎡)', line):
            result["표제부"]["건물내역"] = line.strip()
            break

    # 갑구, 을구 섹션 파싱
    current_section = None
    gap_gu_texts = []
    eul_gu_texts = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "갑구" in line or "갑 구" in line:
            current_section = "갑구"
            continue
        elif "을구" in line or "을 구" in line:
            current_section = "을구"
            continue

        if current_section == "갑구" and line and "순위번호" not in line and "등기목적" not in line:
            gap_gu_texts.append(line)
        elif current_section == "을구" and line and "순위번호" not in line and "등기목적" not in line:
            eul_gu_texts.append(line)

    result["갑구"] = gap_gu_texts if gap_gu_texts else ["데이터 없음"]
    result["을구"] = eul_gu_texts if eul_gu_texts else ["데이터 없음"]

    return result


def is_register_document(file_path):
    """PDF가 등기부등본인지 확인하는 함수"""
    try:
        # 텍스트 기반 PDF 확인
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:3]:  # 처음 3페이지만 확인
                text = page.extract_text()
                if text and "등기사항" in text:
                    return True
        
        # 이미지 기반 PDF인 경우 OCR로 확인
        with fitz.open(file_path) as doc:
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)  # 낮은 해상도로 빠르게 확인
                image = pixmap_to_bgr(pix)
                
                try:
                    response = ocr_google_vision(image)
                    if response.full_text_annotation:
                        page_text = response.full_text_annotation.text
                        if "등기사항" in page_text:
                            return True
                except:
                    continue
                    
        return False
    except Exception as e:
        print(f"문서 유형 확인 중 오류: {e}")
        return False


def extract_all_real_estate_info(file_path):
    # 먼저 문서가 등기부등본인지 확인
    if not is_register_document(file_path):
        raise ValueError("등기부등본 PDF 파일이 아닙니다.")
    
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
            # OCR 텍스트 파싱
            ocr_result = parse_ocr_text_for_registration(all_text_content)

            # 결과 병합
            for key in ["표제부", "발행일", "갑구", "을구"]:
                if ocr_result[key]:
                    result[key] = ocr_result[key]
        else:
            print("OCR에서 텍스트를 추출하지 못했습니다.")
            all_text_content = ""
    else:
        print("텍스트 기반 PDF로 감지됨. 일반 추출을 진행합니다...")
        # 기존 pdfplumber 기반 처리
        result = extract_text_based_pdf(file_path, result)
        all_text_content = extract_all_text_from_pdf(file_path)

    # 법적 상태 확인
    check_legal_status(all_text_content, result["법적상태"])

    return result


def extract_text_based_pdf(file_path, result):
    """텍스트 기반 PDF 처리 (기존 로직)"""
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
                        for j, cell in enumerate(row):
                            if cell and str(cell).strip():
                                clean_cell = str(cell).replace(
                                    "\n", " ").strip()
                                header = header_row[j] if j < len(
                                    header_row) else f"컬럼{j}"
                                row_data.append(
                                    {str(header).strip(): clean_cell})

                                if str(header).strip() == "권리자 및 기타사항":
                                    text = clean_cell
                                    if current_section == "갑구":
                                        owner = re.search(
                                            r"소유자\s+([^\s]+)", text)
                                        price = re.search(
                                            r"거래가액\s*(금[\d,]+원)", text)
                                        if owner:
                                            row_data.append(
                                                {"소유자명": owner.group(1)})
                                        if price:
                                            row_data.append(
                                                {"거래가액": price.group(1)})
                                    elif current_section == "을구":
                                        max_amt = re.search(
                                            r"채권최고액\s*(금[\d,]+원)", text)
                                        if max_amt:
                                            row_data.append(
                                                {"채권최고액": max_amt.group(1)})
                                        debtor = re.search(
                                            r"채무자\s+([^\s]+)\s+(.+?)\s+근저당권자", text)
                                        if debtor:
                                            row_data.append(
                                                {"채무자": debtor.group(1).strip()})
                                            row_data.append(
                                                {"채무자주소": debtor.group(2).strip()})
                                        creditor = re.search(
                                            r"근저당권자\s+([^\d\n]+?)\s+(\d{6,}-\d{6,})?\s*(서울.*)", text)
                                        if creditor:
                                            row_data.append(
                                                {"근저당권자": creditor.group(1).strip()})
                                            row_data.append(
                                                {"근저당권자주소": creditor.group(3).strip()})

                        if row_data:
                            if current_section == "갑구":
                                gap_gu_data.append(row_data)
                            elif current_section == "을구":
                                eul_gu_data.append(row_data)

    result["갑구"] = gap_gu_data if gap_gu_data else gap_gu_raw_texts or ["데이터 없음"]
    result["을구"] = eul_gu_data if eul_gu_data else eul_gu_raw_texts or ["데이터 없음"]

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
    logger = logging.getLogger(__name__)

    # 가압류 관련 키워드
    seizure_keywords = ["가압류", "가압류등기", "가압류신청", "가압류명령", "가압류결정"]
    seizure_negative = ["가압류해제", "가압류취소", "가압류말소"]

    # 경매 관련 키워드
    auction_keywords = ["경매", "경매개시", "경매신청", "경매개시결정", "강제경매", "임의경매"]
    auction_negative = ["경매취소", "경매말소"]

    # 소송 관련 키워드
    lawsuit_keywords = ["소송", "민사소송",
                        "소유권이전등기청구", "소유권이전등기말소", "소유권확인", "손해배상"]
    lawsuit_negative = ["소송취소", "소송말소"]

    # 압류 관련 키워드
    attachment_keywords = ["압류", "압류등기", "압류신청", "압류명령", "압류결정", "강제집행"]
    attachment_negative = ["압류취소", "압류말소"]

    # 키워드 검사
    for keyword in seizure_keywords:
        if keyword in text_content:
            # 부정 표현 확인
            if not any(neg in text_content for neg in seizure_negative):
                legal_status["가압류_여부"] = True
                logger.info(f"가압류 키워드 발견: {keyword}")
                break

    for keyword in auction_keywords:
        if keyword in text_content:
            # 부정 표현 확인
            if not any(neg in text_content for neg in auction_negative):
                legal_status["경매_여부"] = True
                logger.info(f"경매 키워드 발견: {keyword}")
                break

    for keyword in lawsuit_keywords:
        if keyword in text_content:
            # 부정 표현 확인
            if not any(neg in text_content for neg in lawsuit_negative):
                legal_status["소송_여부"] = True
                logger.info(f"소송 키워드 발견: {keyword}")
                break

    for keyword in attachment_keywords:
        if keyword in text_content:
            # 부정 표현 확인
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
                            # 줄바꿈을 띄어쓰기로 변경
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
            # 줄바꿈을 띄어쓰기로 변경
            extracted_data["건물번호"] = building_info["building_number"].replace(
                "\n", " ").replace("  ", " ").strip()

        if building_info["building_detail"] and not extracted_data["건물내역"]:
            # 줄바꿈을 띄어쓰기로 변경
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="등기부등본 PDF 통합 정보 추출기")
    parser.add_argument("file", type=str, help="PDF 파일 경로")
    args = parser.parse_args()

    data = extract_all_real_estate_info(args.file)
    output_dir = "C:/LLM/data/output/registration_json"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_등기부통합.json")
    save_json(data, output_path)
