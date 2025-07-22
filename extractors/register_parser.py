import pdfplumber
import re
import json
import os
from ..config.logger_config import get_logger

logger = get_logger(__name__)


def extract_all_real_estate_info(file_path):
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

    gap_gu_data, eul_gu_data = [], []
    gap_gu_raw_texts, eul_gu_raw_texts = [], []
    title_text_blocks = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if not text or len(text.strip()) < 30:
                logger.warning(
                    f"[주의] 페이지 {page+1}에서 텍스트가 너무 적습니다. 이미지 기반 PDF일 수 있음.")
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
