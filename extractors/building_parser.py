import fitz  # PyMuPDF
import re
import json
import os
import sys
import argparse


def extract_building_info_from_crop(pdf_path, last_word="m", output_dir="../data/output/building_json"):
    """건축물대장 PDF에서 정보를 추출하는 함수"""
    doc = fitz.open(pdf_path)
    page = doc[0]  # 항상 첫 번째 페이지

    # 단어들과 위치 정보 추출
    words = page.get_text("words")
    first, last = None, None

    for word in words:
        word_text = word[4]  # 텍스트는 인덱스 4
        if word_text == "건물ID" and first is None:
            first = word
        # "m" 또는 "숫자m" 패턴 찾기
        if (word_text == last_word or
                (last_word == "m" and (word_text == "m" or re.match(r'\d+\.?\d*m$', word_text)))):
            last = word

    if not first or not last:
        doc.close()
        return None

    # 박스 좌표 계산
    x0 = min(first[0], last[0])
    y0 = min(first[1], last[1])
    x1 = max(first[2], last[2])
    y1 = max(first[3], last[3])

    # 크롭 영역 정의
    crop_rect = fitz.Rect(x0, y0, x1, y1)

    # 크롭된 영역에서 텍스트 추출
    cropped_text = page.get_textbox(crop_rect)

    # 정보 추출
    basic_info = extract_basic_info_from_text(cropped_text)

    # 전체 PDF에서 사용승인일과 위반건축물 여부 찾기
    all_pages_text = ""
    for page_num in range(len(doc)):
        all_pages_text += doc[page_num].get_text() + "\n"

    approval_date = find_approval_date(all_pages_text)
    violation_status = find_violation_status(all_pages_text)

    # 최종 결과
    result = {
        "대지위치": basic_info.get("대지위치", ""),
        "지번": basic_info.get("지번", ""),
        "도로명주소": basic_info.get("도로명주소", ""),
        "연면적": basic_info.get("연면적", None),
        "층수": basic_info.get("층수", None),
        "용도": basic_info.get("용도", []),
        "사용승인일": approval_date,
        "위반건축물여부": violation_status
    }

    # 최종 결과 출력 제거
    # print("\n" + "="*50)
    # print("추출된 건축물 정보")
    # print("="*50)
    # for key, value in result.items():
    #     if key == "용도" and isinstance(value, list):
    #         print(f"{key}: {', '.join(value) if value else '정보없음'}")
    #     else:
    #         print(f"{key}: {value if value is not None and value != '' else '정보없음'}")
    # print("="*50)

    # JSON 파일로 저장
    save_to_json(result, pdf_path, output_dir)

    doc.close()
    return result


def save_to_json(result, pdf_path, output_dir="../data/output/building_json"):
    """추출된 정보를 JSON 파일로 저장"""
    # PDF 파일명에서 확장자 제거하고 JSON 파일명 생성
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    json_filename = f"{base_name}_추출결과.json"

    # 절대 경로로 변환
    output_dir = os.path.abspath(output_dir)

    # 저장할 디렉토리 생성 (중간 디렉토리도 모두 생성)
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, json_filename)

    # JSON 형태로 변환 (연면적이 None이면 문자열로 변환)
    json_result = {}
    for key, value in result.items():
        if key == "연면적" and value is None:
            json_result[key] = "정보없음"
        elif key == "층수" and value is None:
            json_result[key] = "정보없음"
        elif key == "용도" and isinstance(value, list):
            json_result[key] = value if value else ["정보없음"]
        else:
            json_result[key] = value if value != "" else "정보없음"

    # JSON 파일로 저장 (저장 메시지도 제거)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)


def extract_basic_info_from_text(text):
    """크롭된 텍스트에서 기본 정보 추출"""
    info = {}
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # 정보 추출
    for i, line in enumerate(lines):
        extract_location(lines, i, line, info)
        extract_jibun(lines, i, line, info)
        extract_road_address(lines, i, line, info)
        extract_floor_area(lines, i, line, info)
        extract_floors(lines, i, line, info)
        extract_usage(lines, i, line, info)

    # 용도를 찾지 못했다면 다시 한 번 더 찾기
    if "용도" not in info:
        for i, line in enumerate(lines):
            if any(keyword in line for keyword in ["시설", "주택", "오피스텔", "상가", "교육연구"]):
                if "," in line:
                    usages = [u.strip() for u in line.split(",") if u.strip()]
                else:
                    usages = [line.strip()]

                if usages:
                    info["용도"] = usages
                    break

    # 연면적 후보 선택
    select_best_floor_area(info)
    return info


def extract_location(lines, i, line, info):
    """대지위치 찾기"""
    if "대지위치" in line:
        location_pattern = r'대지위치\s*([^지번도로명]*?)(?:지번|도로명|$)'
        match = re.search(location_pattern, line)
        if match and match.group(1).strip():
            location = match.group(1).strip()
            info["대지위치"] = location
        elif i + 1 < len(lines):
            for j in range(i + 1, min(i + 3, len(lines))):
                next_line = lines[j]
                if "도로명주소" in next_line or "지번" in next_line:
                    break
                if any(keyword in next_line for keyword in ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]):
                    info["대지위치"] = next_line
                    break


def extract_jibun(lines, i, line, info):
    """지번 찾기"""
    if "지번" in line:
        # 같은 줄에서 찾기
        jibun_pattern = r'지번\s+([^\s]+)'
        match = re.search(jibun_pattern, line)
        if match and match.group(1).strip():
            jibun = match.group(1).strip()
            info["지번"] = jibun
        else:
            # 다음 줄들을 확인 (최대 5줄)
            for k in range(i + 1, min(i + 6, len(lines))):
                check_line = lines[k].strip()

                # 도로명주소와 함께 있는 경우
                if "도로명주소" in check_line:
                    jibun_match = re.match(r'^([\d-]+)\s+도로명주소', check_line)
                    if jibun_match:
                        jibun = jibun_match.group(1).strip()
                        info["지번"] = jibun
                        break
                    else:
                        continue

                # ※ 기호나 긴 줄은 건너뛰기
                if "※" in check_line or len(check_line) > 15:
                    continue

                # 숫자 패턴인 경우
                if check_line and (check_line.isdigit() or re.match(r'^[\d-]+$', check_line)):
                    info["지번"] = check_line
                    break


def extract_road_address(lines, i, line, info):
    """도로명주소 찾기"""
    if "도로명주소" in line:
        address_pattern = r'도로명주소\s*(.*?\([^)]*\))'
        match = re.search(address_pattern, line)
        if match:
            address = match.group(1).strip()
            info["도로명주소"] = address
        elif i + 1 < len(lines):
            next_line = lines[i + 1]
            if "(" in next_line and ")" in next_line:
                info["도로명주소"] = next_line


def extract_floor_area(lines, i, line, info):
    """연면적 찾기"""
    if "연면적" in line and "용적률" not in line and "산정용" not in line:
        # 같은 줄에서 찾기
        area_patterns = [
            r'연면적\s*([\d,]+\.?\d*)\s*㎡',
            r'연면적.*?([\d,]+\.?\d*)\s*㎡'
        ]

        found = False
        for pattern in area_patterns:
            match = re.search(pattern, line)
            if match:
                area_str = match.group(1).replace(',', '').replace(' ', '')
                try:
                    area = float(area_str)
                    info["연면적"] = area
                    found = True
                    break
                except ValueError:
                    continue

        # 다음 줄들에서 찾기
        if not found:
            for j in range(i + 1, min(i + 15, len(lines))):
                next_line = lines[j]

                # 건축면적/대지면적 건너뛰기
                if any(keyword in next_line for keyword in ["건축면적", "※건축면적", "대지면적", "※대지면적"]):
                    continue

                # 다른 항목 시작시 중단
                if any(keyword in next_line for keyword in ["용적률", "건폐율", "주용도"]):
                    break

                # ㎡ 포함 숫자 찾기
                if "㎡" in next_line and any(c.isdigit() for c in next_line):
                    area_matches = re.findall(r'([\d,]+\.?\d*)\s*㎡', next_line)

                    for area_str in area_matches:
                        try:
                            area = float(area_str.replace(',', ''))

                            if area >= 500000:
                                info["연면적"] = area
                                break
                            elif area >= 100:
                                if "연면적_후보들" not in info:
                                    info["연면적_후보들"] = []
                                info["연면적_후보들"].append(area)
                        except ValueError:
                            continue


def extract_floors(lines, i, line, info):
    """층수 찾기"""
    if any(keyword in line for keyword in ["층수", "지하", "지상"]):
        floor_patterns = [
            r'지하:\s*(\d+)층.*?지상:\s*(\d+)층',
            r'지하\s*(\d+)층.*?지상\s*(\d+)층',
            r'(\d+)/(\d+)',
            r'지하(\d+)층\s*지상(\d+)층'
        ]

        found = False
        for pattern in floor_patterns:
            match = re.search(pattern, line)
            if match:
                basement, above_ground = match.groups()
                info["층수"] = f"지하{basement}층/지상{above_ground}층"
                found = True
                break

        if not found:
            for j in range(i + 1, min(i + 3, len(lines))):
                next_line = lines[j]
                for pattern in floor_patterns:
                    match = re.search(pattern, next_line)
                    if match:
                        basement, above_ground = match.groups()
                        info["층수"] = f"지하{basement}층/지상{above_ground}층"
                        found = True
                        break
                if found:
                    break


def extract_usage(lines, i, line, info):
    """주용도 찾기"""
    if "주용도" in line:
        usage_pattern = r'주용도\s*(.*?)(?:층수|높이|$)'
        match = re.search(usage_pattern, line)
        if match and match.group(1).strip():
            usage_text = match.group(1).strip()
            if ',' in usage_text:
                usages = [u.strip()
                          for u in usage_text.split(',') if u.strip()]
            else:
                usages = [usage_text] if usage_text else []

            if usages:
                info["용도"] = usages
        else:
            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j]

                # 용도 관련 키워드가 있는지 확인
                if any(keyword in next_line for keyword in ["시설", "주택", "오피스텔", "상가", "교육연구"]):
                    # 쉼표로 구분된 여러 용도 처리
                    if "," in next_line:
                        usages = [u.strip()
                                  for u in next_line.split(",") if u.strip()]
                    else:
                        usages = [next_line.strip()]

                    if usages:
                        info["용도"] = usages
                        break

                # 용도를 찾지 못했고, 명확히 다른 항목이면 중단
                if next_line in ["층수"] or any(keyword in next_line for keyword in ["※건폐율", "※용적률", "높이"]):
                    break


def select_best_floor_area(info):
    """연면적 후보 중 최적값 선택"""
    if "연면적_후보들" in info and "연면적" not in info:
        candidates = info["연면적_후보들"]

        if len(candidates) >= 2:
            sorted_candidates = sorted(candidates, reverse=True)
            if sorted_candidates[0] > sorted_candidates[1] * 10:
                info["연면적"] = sorted_candidates[1]
            else:
                info["연면적"] = sorted_candidates[0]
        elif candidates:
            info["연면적"] = candidates[0]

        if "연면적_후보들" in info:
            del info["연면적_후보들"]


def find_approval_date(full_text):
    """사용승인일 찾기"""
    patterns = [
        r'사용승인일\s*(\d{4}\.\d{1,2}\.\d{1,2})',
        r'사용승인\s*(\d{4}\.\d{1,2}\.\d{1,2})',
        r'승인일\s*(\d{4}\.\d{1,2}\.\d{1,2})'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, full_text)
        if matches:
            return matches[0]

    lines = full_text.split('\n')
    for i, line in enumerate(lines):
        if "사용승인" in line:
            context = ' '.join(lines[max(0, i-1):i+2])
            date_match = re.search(r'(\d{4}\.\d{1,2}\.\d{1,2})', context)
            if date_match:
                return date_match.group(1)

    return ""


def find_violation_status(full_text):
    """위반건축물 여부 확인"""
    if any(keyword in full_text for keyword in ["위반", "불법", "무허가"]):
        return "예"
    elif any(keyword in full_text for keyword in ["사용승인", "준공", "허가"]):
        return "아니오"
    else:
        return "확인불가"


def main():
    """메인 함수 - 커맨드라인 인터페이스"""
    parser = argparse.ArgumentParser(
        description='건축물대장 PDF에서 정보를 추출합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
사용 예시:
  python building_parser.py "건축물대장.pdf"
  python building_parser.py "건축물대장.pdf" --last-word "2m"
  python building_parser.py "data/건축물대장.pdf" --output-dir "results"
        '''
    )

    parser.add_argument('pdf_path', help='분석할 PDF 파일 경로')
    parser.add_argument('--last-word', '-l', default='m',
                        help='크롭 범위의 끝 단어 (기본값: m)')
    parser.add_argument('--output-dir', '-o', default='../data/output/building_json',
                        help='결과 저장 디렉토리 (기본값: ../data/output/building_json)')

    args = parser.parse_args()

    # PDF 파일 존재 확인
    if not os.path.exists(args.pdf_path):
        print(f"❌ 오류: 파일을 찾을 수 없습니다 - {args.pdf_path}")
        sys.exit(1)

    try:
        # 정보 추출 실행
        result = extract_building_info_from_crop(
            args.pdf_path,
            last_word=args.last_word,
            output_dir=args.output_dir
        )

        if result:
            pass
        else:
            print("❌ 분석 중 오류가 발생했습니다.")
            sys.exit(1)

    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        sys.exit(1)


# 사용 예시
if __name__ == "__main__":
    main()
