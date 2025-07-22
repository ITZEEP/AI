# 텍스트 기반 및 이미지 기반 계약서 특약사항 추출 모듈 -> main + main2
import pdfplumber
import re
import os
import sys
import json
import numpy as np
import fitz  # PyMuPDF
import cv2
from typing import Optional, List, Tuple
from google.cloud import vision
from dotenv import load_dotenv
from datetime import datetime

# ✅ 환경 변수 및 Vision API 클라이언트 설정
load_dotenv()
json_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if not json_path or not os.path.exists(json_path):
    raise RuntimeError(
        "환경 변수 GOOGLE_APPLICATION_CREDENTIALS가 없거나 경로가 잘못되었습니다.")
vision_client = vision.ImageAnnotatorClient()

# ✅ 텍스트 기반 PDF 특약사항 추출


def extract_special_terms_text_pdf(pdf_path: str) -> str:
    buffer = []
    extracting = False
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split('\n'):
                clean_line = line.strip()
                if '[특약사항]' in clean_line:
                    extracting = True
                    continue
                if extracting:
                    if re.match(r"^-?\s*\d+\s*/\s*\d+\s*-?$", clean_line):
                        continue
                    if '본 계약을 증명하기 위하여' in clean_line:
                        extracting = False
                        break
                    buffer.append(clean_line)
    return '\n'.join(buffer).strip()

# ✅ 이미지 기반 PDF 특약사항 추출


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
    response = vision_client.document_text_detection(image=image)
    if response.error.message:
        raise RuntimeError(f"Google Vision API 오류: {response.error.message}")
    result = []
    if response.full_text_annotation:
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = ''.join([s.text for s in word.symbols])
                        bounding_box = [(v.x, v.y)
                                        for v in word.bounding_box.vertices]
                        result.append((bounding_box, word_text))
    return result


def find_coordinate_markers_debug(ocr_results: List[Tuple[List[Tuple[int, int]], str]]) -> Tuple[float, float]:
    special_terms_y = None
    end_date_y = None
    for bounding_box, text in ocr_results:
        text_clean = text.strip()
        if re.search(r'특약|특별.*약관', text_clean):
            box_y = sum([v[1] for v in bounding_box]) / 4
            special_terms_y = box_y
            print(f"특약사항 Y좌표: {box_y:.2f} → 텍스트: '{text_clean}'")
        if re.search(r'기명', text_clean):
            box_y = sum([v[1] for v in bounding_box]) / 4 - 20
            end_date_y = box_y
            print(f"종료 기준 Y좌표: {box_y:.2f} → 텍스트: '{text_clean}'")
    if special_terms_y and end_date_y:
        print(f"\n📍 최종 추출 범위: {special_terms_y:.2f} ~ {end_date_y:.2f}\n")
    else:
        print("\n특약사항 또는 종료 기준 좌표를 찾지 못했습니다.\n")
    return special_terms_y, end_date_y


def extract_text_between_coordinates(ocr_results, start_y, end_y):
    filtered = []
    for box, text in ocr_results:
        center_y = sum([v[1] for v in box]) / 4
        if start_y <= center_y <= end_y:
            filtered.append((box[0][0], center_y, text))
    lines = {}
    for x, y, text in filtered:
        key = round(y / 20) * 20
        lines.setdefault(key, []).append((x, text))
    result = []
    for y in sorted(lines):
        result.append(" ".join([t for x, t in sorted(lines[y])]))
    return "\n".join(result)


def extract_special_terms_image_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        image = pixmap_to_bgr(pix)
        ocr_results = ocr_google_vision(image)
        if not ocr_results:
            continue
        start_y, end_y = find_coordinate_markers_debug(ocr_results)
        if not start_y or not end_y:
            continue
        text = extract_text_between_coordinates(ocr_results, start_y, end_y)
        if text:
            return text
    return "[특약사항 추출 실패]"

# 최종 자동 분기 함수


def extract_special_terms(pdf_path):
    """계약서 특약사항 추출 후 딕셔너리 반환"""
    result = {
        "file_name": os.path.basename(pdf_path),
        "extracted_at": datetime.now().isoformat(),
        "source": "text",
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = pdf.pages[0].extract_text()
            if text and len(text.strip()) > 100:
                extracted = extract_special_terms_text_pdf(pdf_path)
                result["source"] = "text"
            else:
                extracted = extract_special_terms_image_pdf(pdf_path)
                result["source"] = "image"
            result["special_terms"] = extracted.strip()
    except Exception as e:
        result["error"] = str(e)

    return result


def save_json(output_dict, output_path):
    """JSON 파일로 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)


# ✅ 실행 예시
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="계약서 특약사항 추출기")
    parser.add_argument("file", type=str, help="계약서 PDF 파일 경로")
    args = parser.parse_args()

    data = extract_special_terms(args.file)

    output_dir = "../data/output/contract_json"
    base_name = os.path.splitext(os.path.basename(args.file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_특약.json")

    save_json(data, output_path)
    print(f"[✔] 저장 완료: {output_path}")
