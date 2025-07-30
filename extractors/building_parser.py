
import fitz  # PyMuPDF
import cv2
import numpy as np
from google.cloud import vision
import re
import json
import os
import sys
import argparse
from dotenv import load_dotenv
from typing import Dict, List, Optional

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.path_resolver import get_google_credentials_path


class BuildingInfoExtractor:
    """건축물대장 정보 추출 클래스 - 텍스트 기반 및 이미지 기반 PDF 모두 처리"""

    def __init__(self):
        """환경 변수 및 Google Vision API 클라이언트 초기화"""
        load_dotenv()
        
        # 상대 경로를 절대 경로로 변환
        try:
            json_path = get_google_credentials_path()
            # 환경 변수를 절대 경로로 업데이트
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path
        except (ValueError, FileNotFoundError) as e:
            raise RuntimeError(f"Google Cloud 인증 설정 오류: {e}")
        
        # 클라이언트를 None으로 초기화하고 필요할 때 생성
        self._vision_client = None

    def get_vision_client(self):
        if self._vision_client is None:
            self._vision_client = vision.ImageAnnotatorClient()
        return self._vision_client

    def pixmap_to_bgr(self, pix) -> np.ndarray:
        """PyMuPDF pixmap을 OpenCV BGR 이미지로 변환"""
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def is_building_document(self, doc) -> bool:
        """PDF가 건축물대장인지 확인하는 함수"""
        try:
            # 처음 3페이지만 확인
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                
                # 텍스트 기반 PDF인 경우
                text = page.get_text()
                if text and "건축물대장" in text:
                    return True
                
                # 이미지 기반 PDF인 경우 OCR 수행
                pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))  # 낮은 해상도로 빠르게 확인
                img = self.pixmap_to_bgr(pix)
                
                try:
                    words = self.ocr_google_vision(img)
                    # OCR 결과에서 "건축물대장" 찾기
                    for word in words:
                        if "건축물대장" in word[4]:
                            return True
                except Exception as ocr_error:
                    # OCR 실패시 텍스트만으로 판단
                    print(f"OCR 처리 중 오류 (페이지 {page_num + 1}): {ocr_error}")
                    continue
                    
            return False
        except Exception as e:
            print(f"문서 유형 확인 중 오류: {e}")
            return False

    def ocr_google_vision(self, image_np: np.ndarray) -> List[tuple]:
        """Google Vision API를 사용한 OCR - 단어별 위치 정보와 함께 반환"""
        try:
            # Vision 클라이언트 가져오기
            vision_client = self.get_vision_client()

            success, encoded_image = cv2.imencode('.png', image_np)
            if not success:
                raise RuntimeError("이미지 인코딩 실패")

            image = vision.Image(content=encoded_image.tobytes())
            response = vision_client.document_text_detection(image=image)

            if response.error.message:
                raise RuntimeError(
                    f"Google Vision API 오류: {response.error.message}")

            result = []
            if response.full_text_annotation:
                for page in response.full_text_annotation.pages:
                    for block in page.blocks:
                        for paragraph in block.paragraphs:
                            for word in paragraph.words:
                                word_text = ''.join(
                                    [s.text for s in word.symbols])
                                vertices = word.bounding_box.vertices
                                x0 = min(v.x for v in vertices)
                                y0 = min(v.y for v in vertices)
                                x1 = max(v.x for v in vertices)
                                y1 = max(v.y for v in vertices)
                                result.append((x0, y0, x1, y1, word_text))

            print(f"OCR 결과: {len(result)}개 단어 추출")
            return result
        except Exception as e:
            print(f"OCR 처리 중 오류: {e}")
            raise

    def is_text_based_pdf(self, pdf_path: str) -> bool:
        """PDF가 텍스트 기반인지 확인"""
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            text = page.get_text()
            doc.close()

            print(f"추출된 텍스트 길이: {len(text)}")

            if text and len(text.strip()) > 100 and any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in text):
                return True
            return False
        except Exception:
            return False

    def extract_building_info_from_crop(self, pdf_path: str, last_word: str = "m",
                                        output_dir: str = "../data/output/building_json") -> Optional[Dict]:
        """건축물대장 PDF에서 정보를 추출하는 함수"""
        with fitz.open(pdf_path) as doc:
            # 먼저 문서가 건축물대장인지 확인
            if not self.is_building_document(doc):
                raise ValueError("건축물대장 PDF 파일이 아닙니다.")
            
            if self.is_text_based_pdf(pdf_path):
                print("텍스트 기반 PDF 처리 중...")
                result = self.extract_text_based_pdf(doc, last_word)
            else:
                print("이미지 기반 PDF, Google Vision OCR 처리 중...")
                result = self.extract_image_based_pdf_improved(doc, last_word)

            if result:
                self.save_to_json(result, pdf_path, output_dir)

            return result

    def extract_text_based_pdf(self, doc, last_word: str) -> Optional[Dict]:
        """텍스트 기반 PDF 처리 - 기존 로직 사용하되 연면적 수정"""
        page = doc[0]
        words = page.get_text("words")
        first, last = None, None

        for word in words:
            word_text = word[4]
            if word_text == "건물ID" and first is None:
                first = word
            if (word_text == last_word or
                    (last_word == "m" and (word_text == "m" or re.match(r'\d+\.?\d*m$', word_text)))):
                last = word

        if not first or not last:
            print("ERROR 크롭 영역을 설정할 수 없습니다.")
            return None

        # 박스 좌표 계산
        x0 = min(first[0], last[0])
        y0 = min(first[1], last[1])
        x1 = max(first[2], last[2])
        y1 = max(first[3], last[3])

        crop_rect = fitz.Rect(x0, y0, x1, y1)
        cropped_text = page.get_textbox(crop_rect)

        # 기본 정보 추출 - 연면적 수정된 함수 사용
        basic_info = self.extract_basic_info_from_text_fixed(cropped_text)

        # 전체 PDF에서 사용승인일과 위반건축물 여부 찾기
        all_pages_text = ""
        for page_num in range(len(doc)):
            all_pages_text += doc[page_num].get_text() + "\n"

        approval_date = self.find_approval_date_improved(all_pages_text)
        violation_status = self.find_violation_status(all_pages_text)

        return {
            "대지위치": basic_info.get("대지위치", ""),
            "지번": basic_info.get("지번", ""),
            "도로명주소": basic_info.get("도로명주소", ""),
            "연면적": basic_info.get("연면적", None),
            "층수": basic_info.get("층수", None),
            "용도": basic_info.get("용도", []),
            "사용승인일": approval_date,
            "위반건축물여부": violation_status
        }

    def extract_image_based_pdf_improved(self, doc, last_word: str) -> Optional[Dict]:
        """개선된 이미지 기반 PDF 처리 - OCR 단어 정보 보존"""
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = self.pixmap_to_bgr(pix)
        words = self.ocr_google_vision(img)

        first, last = None, None

        # 시작점과 끝점 찾기
        for i, word in enumerate(words):
            word_text = word[4]
            if word_text == "■" and first is None:
                if i + 2 < len(words) and words[i+1][4] == "건축물" and words[i+2][4] == "대장":
                    first = word
                    print(f"시작 단어 발견: ■ 건축물 대장")
                    break
            elif word_text == "건축물" and first is None:
                if i + 1 < len(words) and words[i+1][4] == "대장":
                    first = word
                    print(f"시작 단어 발견: 건축물 대장")
                    break

        m_candidates = []
        for word in words:
            word_text = word[4]
            if (word_text == last_word or
                    (last_word == "m" and (word_text == "m" or re.match(r'\d+\.?\d*m$', word_text)))):
                m_candidates.append(word)

        if m_candidates:
            last = max(m_candidates, key=lambda w: (w[3], w[2]))
            print(f"끝 단어 발견: {last[4]}")

        if not first or not last:
            print("ERROR 크롭 영역을 설정할 수 없습니다.")
            return None

        # 크롭 영역 처리
        x0 = min(first[0], last[0]) - 20
        y0 = first[1] + 20
        x1 = max(first[2], last[2]) + 20
        y1 = max(first[3], last[3]) + 20

        cropped_words = []
        for word in words:
            word_x0, word_y0, word_x1, word_y1, word_text = word
            if (word_x0 >= x0 and word_y0 >= y0 and word_x1 <= x1 and word_y1 <= y1):
                cropped_words.append((word_y0, word_x0, word_text))

        cropped_text = self.reconstruct_text_from_ocr(cropped_words)
        basic_info = self.extract_basic_info_from_text_fixed(cropped_text)

        # **개선된 사용승인일 검색 - 모든 페이지의 OCR 단어 정보 수집**
        print("전체 PDF에서 사용승인일 검색 중...")
        all_pages_text = ""
        ocr_words_by_page = {}

        for page_num in range(len(doc)):
            print(f"페이지 {page_num + 1} OCR 처리 중...")
            page_obj = doc[page_num]
            pix = page_obj.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = self.pixmap_to_bgr(pix)
            ocr_words = self.ocr_google_vision(img)

            # OCR 단어 정보 저장
            ocr_words_by_page[page_num + 1] = ocr_words

            # 기존 텍스트 재구성도 유지
            page_words = [(word[1], word[0], word[4]) for word in ocr_words]
            page_words.sort(key=lambda x: x[0])

            lines = []
            current_line_words = []
            current_y = None
            y_threshold = 10

            for y, x, text in page_words:
                if current_y is None or abs(y - current_y) <= y_threshold:
                    current_line_words.append((x, text))
                    if current_y is None:
                        current_y = y
                else:
                    if current_line_words:
                        current_line_words.sort(key=lambda w: w[0])
                        line_text = ' '.join([w[1]
                                             for w in current_line_words])
                        lines.append(line_text)
                    current_line_words = [(x, text)]
                    current_y = y

            if current_line_words:
                current_line_words.sort(key=lambda w: w[0])
                line_text = ' '.join([w[1] for w in current_line_words])
                lines.append(line_text)

            page_text = '\n'.join(lines)
            all_pages_text += f"=== 페이지 {page_num + 1} ===\n{page_text}\n\n"

            # 각 페이지에서 사용승인일 찾기 시도
            if "사용승인일" in page_text or "사용 승인일" in page_text:
                print(f"페이지 {page_num + 1}에서 사용승인일 키워드 발견!")

        # 개선된 사용승인일 검색 사용
        approval_date = self.find_approval_date_improved(
            all_pages_text, ocr_words_by_page)
        violation_status = self.find_violation_status(all_pages_text)

        return {
            "대지위치": basic_info.get("대지위치", ""),
            "지번": basic_info.get("지번", ""),
            "도로명주소": basic_info.get("도로명주소", ""),
            "연면적": basic_info.get("연면적", None),
            "층수": basic_info.get("층수", None),
            "용도": basic_info.get("용도", []),
            "사용승인일": approval_date,
            "위반건축물여부": violation_status
        }

    def find_approval_date_improved(self, full_text, ocr_words_by_page=None):
        """개선된 사용승인일 찾기 - OCR 단어 위치 정보 활용"""
        print("개선된 사용승인일 검색 시작...")

        # 방법 1: 기존 텍스트 기반 검색
        approval_date = self.find_approval_date_text_based(full_text)
        if approval_date:
            return approval_date

        # 방법 2: OCR 단어 위치 기반 검색 (더 정확함)
        if ocr_words_by_page:
            approval_date = self.find_approval_date_ocr_based(
                ocr_words_by_page)
            if approval_date:
                return approval_date

        print("사용승인일을 찾을 수 없음")
        return ""

    def find_approval_date_text_based(self, full_text):
        """텍스트 기반 사용승인일 검색"""
        print("텍스트 기반 사용승인일 검색...")

        lines = full_text.split('\n')

        for i, line in enumerate(lines):
            # 기존 조건에 "개사용승인일" 추가
            if any(keyword in line for keyword in ["사용승인일", "사용 승인일", "개사용승인일", "승인일"]):
                print(f"사용승인일 키워드 발견: {line}")

                # 현재 줄과 주변 줄에서 날짜 찾기 (범위 확대)
                context_lines = lines[max(0, i):i+8]  # 더 넓은 범위
                context = ' '.join(context_lines)

                # 디버깅용 출력
                print(f"검색 컨텍스트 ({len(context_lines)}줄):")
                for idx, ctx_line in enumerate(context_lines):
                    print(f"  {idx}: {ctx_line}")

                # 확장된 날짜 패턴들
                date_patterns = [
                    # 기본 패턴들
                    r'사용\s*승인일[:\s]*(\d{4})\.(\d{1,2})\.(\d{1,2})',
                    r'사용\s*승인일[:\s]*(\d{4})[년.-]\s*(\d{1,2})[월.-]\s*(\d{1,2})',
                    r'사용\s*승인일[:\s]*(\d{4})-(\d{1,2})-(\d{1,2})',
                    r'(\d{4})\.(\d{1,2})\.(\d{1,2})',

                    # 역순 패턴들
                    r'(\d{4})\.(\d{1,2})\.(\d{1,2})\s*사용\s*승인일',
                    r'(\d{4})[년.]\s*(\d{1,2})[월.]\s*(\d{1,2})일?\s*사용\s*승인일',

                    # 띄어쓰기가 있는 패턴들
                    r'사용\s+승인일[:\s]*(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})',
                    r'사용\s+승인일[:\s]*(\d{4})\s+(\d{1,2})\s+(\d{1,2})',

                    # OCR 오인식 대응 패턴들
                    r'사용\s*승인일[^0-9]{0,10}(\d{4})[^0-9]{1,3}(\d{1,2})[^0-9]{1,3}(\d{1,2})',
                    r'(\d{4})[^0-9]{1,3}(\d{1,2})[^0-9]{1,3}(\d{1,2})[^사용승인일]{0,20}사용\s*승인일',
                ]

                for pattern_idx, pattern in enumerate(date_patterns):
                    match = re.search(pattern, context, re.IGNORECASE)
                    if match:
                        groups = match.groups()
                        if len(groups) >= 3:
                            year, month, day = groups[:3]
                            # 날짜 유효성 검사
                            try:
                                int_year = int(year)
                                int_month = int(month)
                                int_day = int(day)

                                if 1900 <= int_year <= 2030 and 1 <= int_month <= 12 and 1 <= int_day <= 31:
                                    result_date = f"{year}.{month.zfill(2)}.{day.zfill(2)}"
                                    print(
                                        f"패턴 {pattern_idx}로 사용승인일 발견: {result_date}")
                                    return result_date
                            except ValueError:
                                continue

        return ""

    def find_approval_date_ocr_based(self, ocr_words_by_page):
        """OCR 단어 위치 기반 사용승인일 검색"""
        print("OCR 단어 위치 기반 사용승인일 검색...")

        for page_num, words in ocr_words_by_page.items():
            print(f"페이지 {page_num} 검색 중...")

            # "사용승인일" 관련 단어들 찾기
            approval_keywords = []

            for i, word in enumerate(words):
                x0, y0, x1, y1, text = word

                # 사용승인일 키워드 찾기 (부분 매칭 포함)
                if any(keyword in text for keyword in ["사용승인일", "사용", "승인일", "승인"]):
                    approval_keywords.append((i, x0, y0, x1, y1, text))
                    print(f"  승인 키워드 발견: '{text}' at ({x0}, {y0})")

            # 사용승인일 키워드 근처의 날짜 찾기
            for keyword_info in approval_keywords:
                kw_idx, kw_x0, kw_y0, kw_x1, kw_y1, kw_text = keyword_info
                print(f"  키워드 '{kw_text}' 주변 날짜 검색...")

                # 키워드 주변 영역 설정 (더 넓게)
                search_area = {
                    'x_min': kw_x0 - 200,
                    'x_max': kw_x1 + 200,
                    'y_min': kw_y0 - 50,
                    'y_max': kw_y1 + 100
                }

                # 주변 단어들 수집
                nearby_words = []
                for i in range(max(0, kw_idx - 20), min(len(words), kw_idx + 20)):
                    word = words[i]
                    wx0, wy0, wx1, wy1, wtext = word

                    # 위치 기반 필터링
                    if (search_area['x_min'] <= wx0 <= search_area['x_max'] and
                            search_area['y_min'] <= wy0 <= search_area['y_max']):
                        nearby_words.append((wy0, wx0, wtext))  # y, x, text

                # 위치별 정렬하여 텍스트 재구성
                nearby_words.sort(key=lambda x: (x[0], x[1]))
                nearby_text = ' '.join([w[2] for w in nearby_words])

                print(f"  주변 텍스트: {nearby_text}")

                # 재구성된 텍스트에서 날짜 찾기
                date_patterns = [
                    r'(\d{4})\.(\d{1,2})\.(\d{1,2})',
                    r'(\d{4})\s+(\d{1,2})\s+(\d{1,2})',
                    r'(\d{4})[년]\s*(\d{1,2})[월]\s*(\d{1,2})',
                ]

                for pattern in date_patterns:
                    matches = re.findall(pattern, nearby_text)
                    for match in matches:
                        year, month, day = match
                        try:
                            int_year = int(year)
                            int_month = int(month)
                            int_day = int(day)

                            if 1900 <= int_year <= 2030 and 1 <= int_month <= 12 and 1 <= int_day <= 31:
                                result_date = f"{year}.{month.zfill(2)}.{day.zfill(2)}"
                                print(f"OCR 위치 기반으로 사용승인일 발견: {result_date}")
                                return result_date
                        except ValueError:
                            continue

        return ""

    def extract_basic_info_from_text_fixed(self, text):
        """수정된 기본 정보 추출 - 연면적과 용도 정확히 추출"""
        info = {}
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        print(f"전체 텍스트:\n{text}")
        print(f"줄별 분리: {lines}")

        # 정보 추출
        for i, line in enumerate(lines):
            self.extract_location(lines, i, line, info)
            self.extract_jibun(lines, i, line, info)
            self.extract_road_address(lines, i, line, info)
            self.extract_floor_area_fixed(lines, i, line, info)  # 수정된 연면적 함수
            self.extract_floors(lines, i, line, info)
            self.extract_usage_improved(lines, i, line, info)  # 개선된 용도 함수

        # 용도를 찾지 못했다면 다시 한 번 더 찾기
        if "용도" not in info or not info["용도"]:
            print("용도를 찾지 못함, 추가 검색 시작...")
            self.extract_usage_fallback(lines, info)

        # 연면적 후보 선택
        self.select_best_floor_area(info)
        return info

    def extract_location(self, lines, i, line, info):
        """대지위치 찾기"""
        if "대지위치" in line or "대지 위치" in line:
            location_pattern = r'(?:대지위치|대지\s*위치)\s*([^지번도로명]*?)(?:지번|도로명|$)'
            match = re.search(location_pattern, line)
            if match and match.group(1).strip():
                location = match.group(1).strip()
                info["대지위치"] = location
                print(f"대지위치 발견: {location}")
            elif i + 1 < len(lines):
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_line = lines[j]
                    if any(keyword in next_line for keyword in ["도로명", "주소", "지번"]):
                        break
                    if any(keyword in next_line for keyword in ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]):
                        info["대지위치"] = next_line
                        print(f"대지위치 발견 (다음줄): {next_line}")
                        break

    def extract_jibun(self, lines, i, line, info):
        """지번 찾기"""
        if "지번" in line:
            jibun_pattern = r'지번\s+([^\s]+)'
            match = re.search(jibun_pattern, line)
            if match and match.group(1).strip():
                jibun = match.group(1).strip()
                if jibun not in ["도로명", "주소", "도로명주소"]:
                    info["지번"] = jibun
                    print(f"지번 발견: {jibun}")
            else:
                for k in range(i + 1, min(i + 6, len(lines))):
                    check_line = lines[k].strip()

                    if any(keyword in check_line for keyword in ["도로명", "주소"]):
                        jibun_match = re.search(
                            r'^([\d-]+)\s+(?:도로명|주소)', check_line)
                        if jibun_match:
                            jibun = jibun_match.group(1).strip()
                            print(f"지번 발견 (패턴): {jibun}")
                            break
                        else:
                            continue

                    if "※" in check_line or len(check_line) > 15:
                        continue

                    if check_line and (check_line.isdigit() or re.match(r'^[\d-]+$', check_line)):
                        info["지번"] = check_line
                        print(f"지번 발견 (숫자): {check_line}")
                        break

    def extract_road_address(self, lines, i, line, info):
        """도로명주소 찾기"""
        if "도로명주소" in line or ("도로명" in line and "주소" in line):
            address_pattern = r'(?:도로명주소|도로명\s*주소)\s*(.*?\([^)]*\))'
            match = re.search(address_pattern, line)
            if match:
                address = match.group(1).strip()
                info["도로명주소"] = address
                print(f"도로명주소 발견: {address}")
            elif i + 1 < len(lines):
                next_line = lines[i + 1]
                if "(" in next_line and ")" in next_line:
                    info["도로명주소"] = next_line
                    print(f"도로명주소 발견 (다음줄): {next_line}")

    def extract_floor_area_fixed(self, lines, i, line, info):
        """수정된 연면적 찾기 - 대지면적과 정확히 구분"""
        if "연면적" in line and "용적률" not in line and "산정용" not in line and "대지면적" not in line:
            print(f"연면적 키워드 발견 줄: {line}")

            # 같은 줄에서 찾기
            area_patterns = [
                r'연면적\s*([\d,]+\.?\d*)\s*㎡',
                r'연면적.*?([\d,]+\.?\d*)\s*㎡',
                r'연면적\s*([\d,]+\.?\d*)\s*m',
                r'연면적.*?([\d,]+\.?\d*)\s*m'
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
                        print(f"연면적 발견: {area}")
                        break
                    except ValueError:
                        continue

            # 다음 줄들에서 찾기
            if not found:
                print("같은 줄에서 못 찾음, 다음 줄들 검색...")
                for j in range(i + 1, min(i + 15, len(lines))):
                    next_line = lines[j]
                    print(f"  검사 줄: {next_line}")

                    # **대지면적은 명시적으로 건너뛰기**
                    if any(keyword in next_line for keyword in ["대지면적", "※대지면적", "대지 면적"]):
                        print(f"  건너뛰기 (대지면적): {next_line}")
                        continue

                    # 건축면적도 건너뛰기
                    if any(keyword in next_line for keyword in ["건축면적", "※건축면적"]):
                        print(f"  건너뛰기 (건축면적): {next_line}")
                        continue

                    # 다른 항목 시작시 중단
                    if any(keyword in next_line for keyword in ["용적률", "건폐율", "주용도", "주구조"]):
                        print(f"  중단 (다른 항목): {next_line}")
                        break

                    # ㎡ 또는 m 포함 숫자 찾기 (OCR에서 m으로 인식될 수 있음)
                    if (("㎡" in next_line or "m²" in next_line or " m " in next_line) and any(c.isdigit() for c in next_line)):
                        area_matches = re.findall(
                            r'([\d,]+\.?\d*)\s*(?:㎡|m²|m)', next_line)
                        print(f"  면적 후보들: {area_matches}")

                        for area_str in area_matches:
                            try:
                                area = float(area_str.replace(',', ''))
                                print(f"    검사 면적: {area}")

                                # **연면적은 일반적으로 큰 값이므로 조건 설정**
                                if area >= 100000:  # 10만㎡ 이상이면 바로 연면적으로 인정
                                    info["연면적"] = area
                                    found = True
                                    print(f"연면적 확정: {area}")
                                    break
                                elif area >= 100:  # 1천㎡ 이상이면 후보로 저장
                                    if "연면적_후보들" not in info:
                                        info["연면적_후보들"] = []
                                    info["연면적_후보들"].append(area)
                                    print(f"연면적 후보 추가: {area}")
                            except ValueError:
                                continue

                        if found:
                            break

    def extract_floors(self, lines, i, line, info):
        """층수 찾기"""
        if any(keyword in line for keyword in ["층수", "지하", "지상"]):
            floor_patterns = [
                r'지하:\s*(\d+)층.*?지상:\s*(\d+)층',
                r'지하\s*(\d+)\s*층.*?지상\s*(\d+)\s*층',
                r'(\d+)/(\d+)',
                r'지하(\d+)층\s*지상(\d+)층'
            ]

            found = False
            for pattern in floor_patterns:
                match = re.search(pattern, line)
                if match:
                    basement, above_ground = match.groups()
                    info["층수"] = f"지하{basement}층/지상{above_ground}층"
                    print(f"층수 발견: 지하{basement}층/지상{above_ground}층")
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
                            print(
                                f"층수 발견 (다음줄): 지하{basement}층/지상{above_ground}층")
                            found = True
                            break
                    if found:
                        break

    def extract_usage_improved(self, lines, i, line, info):
        """개선된 주용도 찾기 - 더 정확한 패턴으로 용도 추출"""
        # "주용도" 키워드 찾기
        if "주용도" in line:
            print(f"주용도 키워드 발견: {line}")

            # 주용도가 포함된 줄과 다음 줄들에서 용도 찾기
            search_lines = lines[i:min(i+5, len(lines))]

            # 모든 검색 줄을 하나로 합쳐서 검색
            combined_text = ' '.join(search_lines)
            print(f"용도 검색 텍스트: {combined_text}")

            # 용도 추출
            usages = self.extract_clean_usages_improved(combined_text)
            if usages:
                info["용도"] = usages
                print(f"용도 발견: {usages}")
                return

        # "용도" 키워드만 있는 경우도 처리
        elif "용도" in line and "층별" not in line and "면적" not in line:
            print(f"용도 키워드 발견: {line}")

            # 용도가 포함된 줄과 다음 줄들에서 용도 찾기
            search_lines = lines[i:min(i+3, len(lines))]
            combined_text = ' '.join(search_lines)
            print(f"용도 검색 텍스트: {combined_text}")

            usages = self.extract_clean_usages_improved(combined_text)
            if usages:
                info["용도"] = usages
                print(f"용도 발견: {usages}")

    def extract_clean_usages_improved(self, text):
        """개선된 용도 추출 함수"""
        print(f"용도 텍스트 분석: {text}")

        usages = []

        # 구체적인 용도 패턴들 - 순서대로 우선순위
        usage_patterns = [
            # 교육시설
            (r'교육연구시설', '교육연구시설'),
            (r'교육\s*연구\s*시설', '교육연구시설'),

            # 근린생활시설
            (r'제\s*(\d+)\s*종\s*근린\s*생활\s*시설(?:\s*\([^)]+\))?',
             lambda m: f'제{m.group(1)}종근린생활시설'),
            (r'제(\d+)종근린생활시설(?:\([^)]+\))?',
             lambda m: f'제{m.group(1)}종근린생활시설'),

            # 주택 관련
            (r'단독\s*주택(?:\s*\([^)]+\))?', '단독주택'),
            (r'연립\s*주택', '연립주택'),
            (r'다세대\s*주택', '다세대주택'),
            (r'아파트', '아파트'),
            (r'오피스텔', '오피스텔'),

            # 기타 시설
            (r'업무\s*시설', '업무시설'),
            (r'판매\s*시설', '판매시설'),
            (r'문화\s*및?\s*집회\s*시설', '문화및집회시설'),
            (r'숙박\s*시설', '숙박시설'),
            (r'관광\s*휴게\s*시설', '관광휴게시설'),
        ]

        # 패턴 매칭
        for pattern, replacement in usage_patterns:
            if callable(replacement):
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    usage = replacement(match)
                    if usage and usage not in usages:
                        usages.append(usage)
                        print(f"용도 패턴 매칭: {usage}")
            else:
                if re.search(pattern, text, re.IGNORECASE):
                    if replacement not in usages:
                        usages.append(replacement)
                        print(f"용도 패턴 매칭: {replacement}")

        # 괄호 안의 세부 용도도 추출 (일반음식점 등)
        bracket_patterns = [
            r'\(([^)]*음식점[^)]*)\)',
            r'\(([^)]*상가[^)]*)\)',
            r'\(([^)]*사무[^)]*)\)',
            r'\(([^)]*점포[^)]*)\)',
        ]

        for pattern in bracket_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                clean_match = match.strip()
                if clean_match and len(clean_match) > 1:
                    # 세부 용도는 별도로 추가하지 않고 주용도에 포함된 것으로 처리
                    print(f"세부 용도 발견: {clean_match}")

        return usages

    def extract_usage_fallback(self, lines, info):
        """용도 추출 실패시 대안 검색 함수"""
        print("용도 대안 검색 중...")

        for i, line in enumerate(lines):
            print(f"  검사 줄 {i}: {line}")

            # 용도 관련 키워드가 포함된 줄 찾기
            if any(keyword in line for keyword in ["시설", "주택", "오피스텔", "상가", "교육연구", "업무", "판매", "문화", "숙박", "근린생활"]):
                print(f"  용도 키워드 발견: {line}")

                # 쉼표로 구분된 경우
                if "," in line:
                    usages = []
                    parts = [u.strip() for u in line.split(",") if u.strip()]
                    for part in parts:
                        if any(keyword in part for keyword in ["시설", "주택", "오피스텔", "상가"]):
                            # 불필요한 부분 제거
                            clean_part = re.sub(r'철근\s*콘크리트\s*구조', '', part)
                            clean_part = re.sub(
                                r'지하\s*\d+\s*층', '', clean_part)
                            clean_part = re.sub(
                                r'지상\s*\d+\s*층', '', clean_part)
                            clean_part = re.sub(
                                r'[\d,]+\.?\d*\s*(?:㎡|m²)', '', clean_part)
                            clean_part = clean_part.strip()

                            if clean_part and len(clean_part) > 2:
                                usages.append(clean_part)
                                print(f"    용도 추가: {clean_part}")

                    if usages:
                        info["용도"] = usages
                        print(f"대안 검색으로 용도 발견: {usages}")
                        return

                # 단일 용도인 경우
                else:
                    # 불필요한 부분 제거
                    clean_line = re.sub(r'철근\s*콘크리트\s*구조', '', line)
                    clean_line = re.sub(r'지하\s*\d+\s*층', '', clean_line)
                    clean_line = re.sub(r'지상\s*\d+\s*층', '', clean_line)
                    clean_line = re.sub(
                        r'[\d,]+\.?\d*\s*(?:㎡|m²)', '', clean_line)
                    clean_line = clean_line.strip()

                    if clean_line and len(clean_line) > 2:
                        # 특정 키워드만 추출
                        if any(keyword in clean_line for keyword in ["시설", "주택", "오피스텔"]):
                            info["용도"] = [clean_line.strip()]
                            print(f"대안 검색으로 단일 용도 발견: {clean_line}")
                            return

        # 마지막 수단: 전체 텍스트에서 패턴 찾기
        all_text = ' '.join(lines)

        # 일반적인 용도 패턴들
        fallback_patterns = [
            r'(교육연구시설)',
            r'(제\d+종근린생활시설)',
            r'(단독주택)',
            r'(다세대주택)',
            r'(연립주택)',
            r'(아파트)',
            r'(오피스텔)',
            r'(업무시설)',
            r'(상업시설)',
            r'(판매시설)'
        ]

        found_usages = []
        for pattern in fallback_patterns:
            matches = re.findall(pattern, all_text)
            for match in matches:
                if match not in found_usages:
                    found_usages.append(match)
                    print(f"패턴 매칭으로 용도 발견: {match}")

        if found_usages:
            info["용도"] = found_usages
            print(f"최종 대안 검색으로 용도 발견: {found_usages}")
        else:
            print("용도를 찾을 수 없음")

    def reconstruct_text_from_ocr(self, cropped_words) -> str:
        """OCR 결과를 줄별로 재구성하여 텍스트 형태로 변환"""
        print(f"크롭된 단어 수: {len(cropped_words)}")

        # y좌표로 정렬
        cropped_words.sort(key=lambda x: x[0])

        # 줄별로 그룹화
        lines = []
        current_line_words = []
        current_y = None
        y_threshold = 8

        for y, x, text in cropped_words:
            if current_y is None or abs(y - current_y) <= y_threshold:
                current_line_words.append((x, text))
                if current_y is None:
                    current_y = y
            else:
                if current_line_words:
                    current_line_words.sort(key=lambda w: w[0])
                    line_text = ' '.join([w[1] for w in current_line_words])
                    lines.append(line_text)

                current_line_words = [(x, text)]
                current_y = y

        if current_line_words:
            current_line_words.sort(key=lambda w: w[0])
            line_text = ' '.join([w[1] for w in current_line_words])
            lines.append(line_text)

        # 텍스트 정리
        processed_lines = []
        for line in lines:
            cleaned_line = re.sub(r'\s+', ' ', line.strip())
            if cleaned_line:
                processed_lines.append(cleaned_line)

        cropped_text = '\n'.join(processed_lines)
        print(f"크롭된 텍스트 줄 수: {len(processed_lines)}")
        return cropped_text

    def select_best_floor_area(self, info):
        """연면적 후보 중 최적값 선택"""
        if "연면적_후보들" in info and "연면적" not in info:
            candidates = info["연면적_후보들"]
            print(f"연면적 후보들: {candidates}")

            if len(candidates) >= 2:
                # 여러 후보가 있으면 가장 큰 값을 연면적으로 선택
                sorted_candidates = sorted(candidates, reverse=True)
                info["연면적"] = sorted_candidates[0]
                print(f"연면적 후보 중 선택: {sorted_candidates[0]}")
            elif candidates:
                info["연면적"] = candidates[0]
                print(f"연면적 후보 중 선택: {candidates[0]}")

            if "연면적_후보들" in info:
                del info["연면적_후보들"]

    def find_violation_status(self, full_text):
        """위반건축물 여부 확인"""
        if any(keyword in full_text for keyword in ["위반", "불법", "무허가"]):
            return "예"
        elif any(keyword in full_text for keyword in ["사용승인", "준공", "허가"]):
            return "아니오"
        else:
            return "확인불가"

    def debug_approval_date_search(self, doc):
        """사용승인일 디버깅용 함수 - 페이지별 상세 분석"""

        for page_num in range(len(doc)):
            print(f"\n=== 페이지 {page_num + 1} 디버깅 ===")
            page_obj = doc[page_num]
            pix = page_obj.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = self.pixmap_to_bgr(pix)
            ocr_words = self.ocr_google_vision(img)

            # 사용승인일 관련 단어들 찾기
            approval_related = []
            date_words = []

            for i, word in enumerate(ocr_words):
                x0, y0, x1, y1, text = word

                # 사용승인일 관련 키워드
                if any(keyword in text.lower() for keyword in ['사용', '승인', '승인일']):
                    approval_related.append((i, text, x0, y0))
                    print(f"승인 관련 단어 {i}: '{text}' at ({x0}, {y0})")

            # 상위 10개만
            print(f"날짜 형태 단어들: {[(w[0], w[1]) for w in date_words[:10]]}")

            # 사용승인일 키워드 주변의 단어들 상세 분석
            for keyword_info in approval_related:
                kw_idx, kw_text, kw_x, kw_y = keyword_info
                print(f"\n키워드 '{kw_text}' 주변 분석:")

                # 주변 20개 단어 출력
                start_idx = max(0, kw_idx - 10)
                end_idx = min(len(ocr_words), kw_idx + 10)

                for i in range(start_idx, end_idx):
                    word = ocr_words[i]
                    marker = " >>> " if i == kw_idx else "     "
                    print(
                        f"{marker}{i}: '{word[4]}' at ({word[0]}, {word[1]})")

            # 줄별 재구성된 텍스트도 출력
            print(f"\n페이지 {page_num + 1} 재구성된 텍스트:")
            page_words = [(word[1], word[0], word[4]) for word in ocr_words]
            page_words.sort(key=lambda x: x[0])

            lines = []
            current_line_words = []
            current_y = None
            y_threshold = 10

            for y, x, text in page_words:
                if current_y is None or abs(y - current_y) <= y_threshold:
                    current_line_words.append((x, text))
                    if current_y is None:
                        current_y = y
                else:
                    if current_line_words:
                        current_line_words.sort(key=lambda w: w[0])
                        line_text = ' '.join([w[1]
                                             for w in current_line_words])
                        lines.append(line_text)
                    current_line_words = [(x, text)]
                    current_y = y

            if current_line_words:
                current_line_words.sort(key=lambda w: w[0])
                line_text = ' '.join([w[1] for w in current_line_words])
                lines.append(line_text)

            for i, line in enumerate(lines):
                if '사용' in line or '승인' in line:
                    print(f">>> 라인 {i}: {line}")
                else:
                    print(f"    라인 {i}: {line}")

    def save_to_json(self, result: Dict, pdf_path: str, output_dir: str = "../data/output/building_json"):
        """추출된 정보를 JSON 파일로 저장"""
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        json_filename = f"{base_name}_추출결과.json"

        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, json_filename)

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

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)

        print(f"OK 결과 저장됨: {json_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='건축물대장 PDF에서 정보를 추출합니다.')
    parser.add_argument('pdf_path', help='분석할 PDF 파일 경로')
    parser.add_argument('--last-word', '-l', default='m',
                        help='크롭 범위의 끝 단어 (기본값: m)')
    parser.add_argument(
        '--output-dir', '-o', default='../data/output/building_json', help='결과 저장 디렉토리')
    parser.add_argument(
        '--debug', '-d', action='store_true', help='디버깅 모드 활성화')

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"ERROR 오류: 파일을 찾을 수 없습니다 - {args.pdf_path}")
        sys.exit(1)

    try:
        extractor = BuildingInfoExtractor()

        # 디버깅 모드
        if args.debug:
            doc = fitz.open(args.pdf_path)
            extractor.debug_approval_date_search(doc)
            doc.close()
            return

        # 일반 실행 모드
        result = extractor.extract_building_info_from_crop(
            args.pdf_path,
            last_word=args.last_word,
            output_dir=args.output_dir
        )

        if result:
            print("OK 건축물대장 분석 완료")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("ERROR 분석 중 오류가 발생했습니다.")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR 오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
