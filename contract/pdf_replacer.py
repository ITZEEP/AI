"""PDF Text Replacer for Contract Generation

계약서 PDF 템플릿의 텍스트를 교체하는 유틸리티 클래스
REST API와 통합하여 사용하도록 최적화
"""

import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.utils import ImageReader
from PyPDF2 import PdfReader, PdfWriter
from PIL import Image
import io
import os
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PDFTextReplacer:
    """PDF 텍스트 교체 클래스 - API 통합용"""
    
    # 기본 설정값
    DEFAULT_FONT_SIZE = 10
    DEFAULT_LINE_HEIGHT = 15
    DEFAULT_BULLET_INDENT = 15
    DEFAULT_WRAP_WIDTH = 400
    
    # 한글 폰트 경로 (고정)
    FONT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'font', 'NanumGothic.ttf'
    )
    
    # 기본 줄바꿈 너비 설정
    DEFAULT_WRAP_WIDTHS = {
        "${special}": 500,   # 특약사항
        "${sy}": 50,         # 연도
        "${sm}": 30,         # 월
        "${sd}": 30,         # 일
        "${ey}": 50,
        "${em}": 30,
        "${ed}": 30,
        "${nowY}": 50,
        "${nowM}": 30,
        "${nowD}": 30,
        "default": 400       # 기본값
    }
    
    def __init__(self):
        """
        PDFTextReplacer 초기화
        """
        self.font_name = None
        self.font_size = self.DEFAULT_FONT_SIZE
        self.line_height = self.DEFAULT_LINE_HEIGHT
        self.bullet_indent = self.DEFAULT_BULLET_INDENT
        
        # 폰트 초기화
        self._initialize_font()
        
    def _initialize_font(self) -> None:
        """
        한글 폰트 초기화
        
        Raises:
            FileNotFoundError: 폰트 파일이 없을 때
            Exception: 폰트 등록 실패 시
        """
        # 폰트 파일 존재 확인
        if not os.path.exists(self.FONT_PATH):
            error_msg = (
                f"한글 폰트 파일을 찾을 수 없습니다: {self.FONT_PATH}\n"
                "NanumGothic.ttf 파일을 data/font/ 디렉토리에 추가해주세요."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 폰트 등록
        try:
            pdfmetrics.registerFont(TTFont('NanumGothic', self.FONT_PATH))
            self.font_name = 'NanumGothic'
            logger.info(f"한글 폰트 등록 성공: {self.FONT_PATH}")
        except Exception as e:
            error_msg = f"한글 폰트 등록 실패: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def remove_text_from_pdf(self, input_pdf: str, output_pdf: str, 
                            search_texts: List[str]) -> Dict[str, List[Dict]]:
        """
        PDF에서 텍스트를 완전히 제거 (Redaction 사용)
        
        Args:
            input_pdf: 입력 PDF 파일 경로
            output_pdf: 출력 PDF 파일 경로
            search_texts: 제거할 텍스트 리스트
            
        Returns:
            제거된 텍스트의 위치 정보 딕셔너리
        """
        positions = {}
        doc = fitz.open(input_pdf)
        
        logger.info(f"PDF에서 텍스트 제거 중: {input_pdf}")
        logger.debug(f"총 페이지 수: {len(doc)}")
        
        for page_num, page in enumerate(doc):
            
            for text in search_texts:
                instances = page.search_for(text)
                
                if instances:
                    if text not in positions:
                        positions[text] = []
                    
                    for inst in instances:
                        # 위치 정보 저장
                        pos_info = {
                            'page': page_num,
                            'x': inst.x0,
                            'y': page.rect.height - inst.y1 + 2,  # ReportLab은 좌하단이 원점
                            'orig_width': inst.width,
                            'orig_height': inst.height,
                            'page_width': page.rect.width,
                            'page_height': page.rect.height
                        }
                        positions[text].append(pos_info)
                        
                        # Redaction으로 텍스트 완전 제거
                        page.add_redact_annot(inst)
                    
                    logger.debug(f"'{text}' → {len(instances)}개 제거")
            
            # Redaction 적용
            page.apply_redactions()
        
        # 텍스트가 제거된 PDF 저장
        doc.save(output_pdf)
        doc.close()
        
        return positions
    
    def wrap_text(self, canvas_obj: canvas.Canvas, text: str, 
                 max_width: float) -> List[str]:
        """
        텍스트를 주어진 너비에 맞게 줄바꿈
        
        Args:
            canvas_obj: Canvas 객체
            text: 줄바꿈할 텍스트
            max_width: 최대 너비
            
        Returns:
            줄바꿈된 텍스트 리스트
        """
        lines = []
        paragraphs = text.split('\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                lines.append("")
                continue
            
            words = paragraph.split()
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                text_width = canvas_obj.stringWidth(test_line, self.font_name, self.font_size)
                
                if text_width <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    
                    # 단어가 너무 길면 문자 단위로 자르기
                    if canvas_obj.stringWidth(word, self.font_name, self.font_size) > max_width:
                        chars = list(word)
                        temp_line = ""
                        for char in chars:
                            if canvas_obj.stringWidth(temp_line + char, self.font_name, self.font_size) <= max_width:
                                temp_line += char
                            else:
                                if temp_line:
                                    lines.append(temp_line)
                                temp_line = char
                        if temp_line:
                            current_line = temp_line
                    else:
                        current_line = word
            
            if current_line:
                lines.append(current_line)
        
        return lines
    
    def wrap_text_with_bullets(self, canvas_obj: canvas.Canvas, text: Union[str, List[str]], 
                              max_width: float) -> List[Dict[str, str]]:
        """
        텍스트를 줄바꿈하면서 불릿 포인트 처리
        
        Args:
            canvas_obj: Canvas 객체
            text: 줄바꿈할 텍스트 또는 리스트
            max_width: 최대 너비
            
        Returns:
            줄바꿈된 텍스트 정보 리스트
        """
        lines = []
        # 리스트인 경우 각 항목을 별도의 paragraph로 처리
        if isinstance(text, list):
            paragraphs = text
        else:
            paragraphs = text.split('\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                lines.append({"type": "empty", "text": ""})
                continue
            
            words = paragraph.split()
            current_line = ""
            paragraph_lines = []
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                text_width = canvas_obj.stringWidth(test_line, self.font_name, self.font_size)
                
                if text_width <= (max_width - self.bullet_indent):
                    current_line = test_line
                else:
                    if current_line:
                        paragraph_lines.append(current_line)
                    
                    if canvas_obj.stringWidth(word, self.font_name, self.font_size) > (max_width - self.bullet_indent):
                        chars = list(word)
                        temp_line = ""
                        for char in chars:
                            if canvas_obj.stringWidth(temp_line + char, self.font_name, self.font_size) <= (max_width - self.bullet_indent):
                                temp_line += char
                            else:
                                if temp_line:
                                    paragraph_lines.append(temp_line)
                                temp_line = char
                        if temp_line:
                            current_line = temp_line
                    else:
                        current_line = word
            
            if current_line:
                paragraph_lines.append(current_line)
            
            # 첫 번째 줄에는 불릿 포인트, 나머지는 들여쓰기
            for i, line in enumerate(paragraph_lines):
                if i == 0:
                    lines.append({"type": "bullet", "text": line})
                else:
                    lines.append({"type": "indent", "text": line})
        
        return lines
    
    def format_list_text(self, text_list: Union[str, List[str]], 
                        separator: str = "\n") -> str:
        """
        리스트를 텍스트로 변환
        
        Args:
            text_list: 텍스트 또는 텍스트 리스트
            separator: 구분자
            
        Returns:
            포맷된 텍스트
        """
        if isinstance(text_list, list):
            return separator.join(text_list)
        return text_list
    
    def process_signature_image(self, image_data: bytes, max_width: float = 100, 
                               max_height: float = 50) -> tuple:
        """
        서명 이미지 처리 및 크기 조정 (비율 유지)
        
        Args:
            image_data: 이미지 바이너리 데이터
            max_width: 최대 너비
            max_height: 최대 높이
            
        Returns:
            (처리된 이미지 바이너리 데이터, 실제 너비, 실제 높이)
        """
        try:
            # PIL Image로 열기
            img = Image.open(io.BytesIO(image_data))
            
            # RGBA로 변환 (투명 배경 지원)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # 원본 비율 계산
            orig_width, orig_height = img.size
            aspect_ratio = orig_width / orig_height
            
            # 비율을 유지하면서 최대 크기에 맞추기
            if orig_width / max_width > orig_height / max_height:
                # 너비가 제한 요소
                new_width = max_width
                new_height = max_width / aspect_ratio
            else:
                # 높이가 제한 요소
                new_height = max_height
                new_width = max_height * aspect_ratio
            
            # 크기 조정
            img.thumbnail((new_width * 2, new_height * 2), Image.Resampling.LANCZOS)
            
            # BytesIO로 저장
            output = io.BytesIO()
            img.save(output, format='PNG')
            output.seek(0)
            
            return output.getvalue(), new_width, new_height
            
        except Exception as e:
            logger.error(f"이미지 처리 실패: {e}")
            # 오류 시 기본값 반환
            return image_data, max_width, max_height
    
    def add_text_to_pdf(self, cleaned_pdf: str, output_pdf: str, 
                       positions: Dict[str, List[Dict]], 
                       replacements: Dict[str, Union[str, List[str]]], 
                       wrap_widths: Optional[Dict[str, float]] = None,
                       images: Optional[Dict[str, bytes]] = None):
        """
        텍스트가 제거된 PDF에 새 텍스트 및 이미지 추가
        
        Args:
            cleaned_pdf: 텍스트가 제거된 PDF 파일
            output_pdf: 최종 출력 PDF 파일
            positions: 텍스트 위치 정보
            replacements: 교체할 텍스트 딕셔너리
            wrap_widths: 줄바꿈 너비 설정
            images: 서명 이미지 딕셔너리 (키: 필드명, 값: 이미지 바이너리)
        """
        
        # 기본 줄바꿈 너비 설정
        if wrap_widths is None:
            wrap_widths = self.DEFAULT_WRAP_WIDTHS
        
        # PDF 읽기
        with open(cleaned_pdf, "rb") as f:
            pdf_reader = PdfReader(f)
            pdf_writer = PdfWriter()
            
            # 각 페이지별로 처리
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                
                # 이 페이지에 추가할 텍스트 수집
                texts_for_page = []
                for old_text, new_text in replacements.items():
                    if old_text in positions:
                        for pos in positions[old_text]:
                            if pos['page'] == page_num:
                                # 리스트는 그대로 유지, 문자열만 포맷
                                if isinstance(new_text, list):
                                    texts_for_page.append((pos, new_text, old_text, True))
                                else:
                                    texts_for_page.append((pos, new_text, old_text, False))
                
                # 텍스트가 있으면 오버레이 생성
                if texts_for_page:
                    # 오버레이 PDF 생성
                    packet = io.BytesIO()
                    can = canvas.Canvas(packet, pagesize=(pos['page_width'], pos['page_height']))
                    
                    # 텍스트 추가
                    can.setFont(self.font_name, self.font_size)
                    can.setFillColorRGB(0, 0, 0)  # 검정색
                    
                    for pos, new_text, old_text, is_list in texts_for_page:
                        # 줄바꿈 너비 결정
                        if old_text in wrap_widths:
                            max_width = wrap_widths[old_text]
                        elif 'default' in wrap_widths:
                            max_width = wrap_widths['default']
                        else:
                            max_width = max(pos['orig_width'] * 1.5, self.DEFAULT_WRAP_WIDTH)
                        
                        logger.debug(f"텍스트 '{old_text[:20]}...' → 줄바꿈 너비: {max_width:.1f}")
                        
                        if is_list:
                            # 리스트 처리 (불릿 포인트)
                            lines = self.wrap_text_with_bullets(can, new_text, max_width)
                            
                            y_position = pos['y']
                            for line_info in lines:
                                if line_info["type"] == "empty":
                                    y_position -= self.line_height
                                elif line_info["type"] == "bullet":
                                    can.drawString(pos['x'], y_position, "•")
                                    # 텍스트가 문자열인지 확인
                                    text_to_draw = str(line_info["text"]) if line_info["text"] else ""
                                    can.drawString(pos['x'] + self.bullet_indent, y_position, text_to_draw)
                                    y_position -= self.line_height
                                elif line_info["type"] == "indent":
                                    # 텍스트가 문자열인지 확인  
                                    text_to_draw = str(line_info["text"]) if line_info["text"] else ""
                                    can.drawString(pos['x'] + self.bullet_indent, y_position, text_to_draw)
                                    y_position -= self.line_height
                        elif isinstance(new_text, str) and len(new_text) > 30:
                            # 일반 긴 텍스트 처리
                            lines = self.wrap_text(can, new_text, max_width)
                            
                            y_position = pos['y']
                            for line in lines:
                                # 텍스트가 문자열인지 확인
                                text_to_draw = str(line) if line else ""
                                can.drawString(pos['x'], y_position, text_to_draw)
                                y_position -= self.line_height
                        elif isinstance(new_text, str):
                            # 짧은 텍스트
                            logger.debug(f"Drawing short text at {pos['x']}, {pos['y']}: {new_text!r} (type: {type(new_text).__name__})")
                            can.drawString(pos['x'], pos['y'], new_text)
                        else:
                            # 예상치 못한 타입 (빈 문자열로 처리)
                            logger.warning(f"Unexpected type for field {old_text}: {type(new_text).__name__} = {new_text!r}")
                            # 리스트면 문자열로 변환
                            if isinstance(new_text, list):
                                logger.warning(f"Converting list to empty string for field {old_text}")
                                # 빈 문자열로 처리 (또는 첫 번째 항목 사용)
                                pass  # 아무것도 그리지 않음
                    
                    # 이미지 추가 (서명 등)
                    if images:
                        for img_key, img_data in images.items():
                            # 이미지 위치를 텍스트 위치에서 찾기
                            img_placeholder = f"${{{img_key}}}"
                            if img_placeholder in positions:
                                for pos in positions[img_placeholder]:
                                    if pos['page'] == page_num:
                                        try:
                                            # 이미지 처리 (비율 유지)
                                            processed_img, actual_width, actual_height = self.process_signature_image(
                                                img_data, 
                                                max_width=100,  # 최대 너비
                                                max_height=50   # 최대 높이
                                            )
                                            img_reader = ImageReader(io.BytesIO(processed_img))
                                            
                                            # 이미지 그리기 (오른쪽 아래 꼭짓점 기준)
                                            # pos['x'], pos['y']가 오른쪽 아래 꼭짓점이 되도록 조정
                                            can.drawImage(img_reader, 
                                                        pos['x']-20,   # x 좌표에서 실제 너비만큼 왼쪽으로
                                                        pos['y']-30,                   # y 좌표 그대로 (이미 하단 기준)
                                                        width=actual_width, 
                                                        height=actual_height,
                                                        preserveAspectRatio=True,
                                                        mask='auto')
                                        except Exception as e:
                                            logger.error(f"이미지 삽입 실패 ({img_key}): {e}")
                    
                    can.save()
                    packet.seek(0)
                    
                    # 오버레이 병합
                    overlay_pdf = PdfReader(packet)
                    page.merge_page(overlay_pdf.pages[0])
                
                pdf_writer.add_page(page)
            
            # 최종 PDF 저장
            with open(output_pdf, "wb") as output_file:
                pdf_writer.write(output_file)
        
        logger.info(f"최종 PDF 생성 완료: {output_pdf}")
    
    def replace_text(self, input_pdf: str, output_pdf: str, 
                    replacements: Dict[str, Union[str, List[str]]], 
                    wrap_widths: Optional[Dict[str, float]] = None,
                    images: Optional[Dict[str, bytes]] = None) -> bool:
        """
        PDF 텍스트 완전 교체 메인 메서드
        
        Args:
            input_pdf: 입력 PDF 파일
            output_pdf: 출력 PDF 파일
            replacements: 교체할 텍스트 딕셔너리
            wrap_widths: 줄바꿈 너비 설정
            images: 서명 이미지 딕셔너리
            
        Returns:
            성공 여부
        """
        logger.info("PDF 텍스트 교체 시작")
        
        try:
            # 텍스트 제거 및 위치 정보 획득
            temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False).name
            positions = self.remove_text_from_pdf(input_pdf, temp_pdf, list(replacements.keys()))
            
            if not positions:
                logger.warning("교체할 텍스트를 찾을 수 없습니다!")
                return False
            
            # 새 텍스트 및 이미지 추가
            logger.info("새 텍스트 및 이미지 추가 중...")
            self.add_text_to_pdf(temp_pdf, output_pdf, positions, replacements, wrap_widths, images)
            
            # 임시 파일 삭제
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)
            
            logger.info("PDF 텍스트 교체 완료")
            return True
            
        except Exception as e:
            logger.error(f"PDF 교체 중 오류 발생: {e}")
            # 임시 파일 정리
            if 'temp_pdf' in locals() and os.path.exists(temp_pdf):
                os.remove(temp_pdf)
            raise
    
    def generate_contract_pdf(self, template_path: str, contract_data: Dict[str, Any], 
                            images: Optional[Dict[str, bytes]] = None) -> bytes:
        """
        계약서 데이터를 받아 PDF 생성 (API 통합용)
        
        Args:
            template_path: PDF 템플릿 파일 경로
            contract_data: 계약서 데이터 (API request body)
            images: 서명 이미지 딕셔너리
            
        Returns:
            생성된 PDF 파일 바이너리 데이터
        """
        # 임시 출력 파일
        output_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        output_path = output_file.name
        output_file.close()
        
        try:
            # 데이터 변환 (API 필드명 → 템플릿 변수명)
            replacements = self._prepare_replacements(contract_data)
            
            # 조건에 따라 이미지 필터링
            filtered_images = {}
            if images:
                # 옵셔널 필드: 받지 않으면 False로 처리 (서명 없이 계약서 생성)
                has_tax_arrears = contract_data.get('hasTaxArrears', False)
                has_prior_fixed_date = contract_data.get('hasPriorFixedDate', False)
                
                # ownerSign1은 미납 세금이 있을 때만 포함
                if has_tax_arrears and 'ownerSign1' in images:
                    filtered_images['ownerSign1'] = images['ownerSign1']
                
                # ownerSign2는 선순위 확정일자가 있을 때만 포함
                if has_prior_fixed_date and 'ownerSign2' in images:
                    filtered_images['ownerSign2'] = images['ownerSign2']
                
                # ownerSign3과 buyerSign1은 항상 포함 (이미지가 있을 때만)
                if 'ownerSign3' in images:
                    filtered_images['ownerSign3'] = images['ownerSign3']
                if 'buyerSign1' in images:
                    filtered_images['buyerSign1'] = images['buyerSign1']
            
            # PDF 생성
            success = self.replace_text(
                input_pdf=template_path,
                output_pdf=output_path,
                replacements=replacements,
                wrap_widths=self.DEFAULT_WRAP_WIDTHS,
                images=filtered_images
            )
            
            if success:
                # 생성된 PDF를 바이너리로 읽기
                with open(output_path, 'rb') as f:
                    pdf_content = f.read()
                return pdf_content
            else:
                raise Exception("PDF 생성 실패")
                
        finally:
            # 임시 파일 삭제
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def _prepare_replacements(self, contract_data: Dict[str, Any]) -> Dict[str, Union[str, List[str]]]:
        """
        API 요청 데이터를 템플릿 변수로 변환
        SaveFinalContractDTO 형식 지원
        
        Args:
            contract_data: API 요청 데이터
            
        Returns:
            템플릿 변수 딕셔너리
        """
        # 디버깅: 입력 데이터 타입 확인
        logger.debug(f"Contract data types: {[(k, type(v).__name__) for k, v in contract_data.items()]}")
        
        # 기본값 설정
        replacements = {}
        
        # leaseType으로 전세/월세 체크박스 설정 (true면 전세, false면 월세)
        lease_type = contract_data.get('leaseType', True)
        replacements.update({
            "${1}": "■" if lease_type else "□",  # 전세 체크박스
            "${2}": "□" if lease_type else "■",  # 월세 체크박스
        })
        
        # hasTaxArrears로 체크박스 설정 (미납 국세/지방세)
        # 옵셔널 필드: 받지 않으면 False (없음)으로 처리
        has_tax_arrears = contract_data.get('hasTaxArrears', False)
        replacements.update({
            "${3}": "■" if has_tax_arrears else "□",  # 있음
            "${4}": "□" if has_tax_arrears else "■",  # 없음
        })
        
        # hasPriorFixedDate로 체크박스 설정 (선순위 확정일자)
        # 옵셔널 필드: 받지 않으면 False (없음)으로 처리
        has_prior_fixed_date = contract_data.get('hasPriorFixedDate', False)
        replacements.update({
            "${5}": "■" if has_prior_fixed_date else "□",  # 있음
            "${6}": "□" if has_prior_fixed_date else "■",  # 없음
        })
        
        # 추가 체크박스 (기본값 설정 - 필요시 조정)
        checkbox4 = contract_data.get('checkbox4', False)
        replacements.update({
            "${7}": "■" if checkbox4 else "□",
            "${8}": "□" if checkbox4 else "■",
        })
        
        # 주소 정보
        replacements.update({
            "${roadAddr}": str(contract_data.get('addr1', '')),  # addr1이 도로명 주소
            "${addr2}": str(contract_data.get('addr2', '')),     # addr2가 임차할 부분 주소
        })
        
        # 면적 정보
        area = contract_data.get('area', 0)
        supply_area = contract_data.get('supplyArea', 0)
        total_floor_area = contract_data.get('totalFloorArea', 0)
        
        # buildingStructure와 purpose를 합쳐서 use 필드에 넣기
        building_structure = contract_data.get('buildingStructure', '철근콘크리트 구조')
        purpose = contract_data.get('purpose', '')
        use_combined = f"{building_structure}/{purpose}" if purpose else building_structure
        
        replacements.update({
            "${site1}": str(supply_area) if supply_area else '',     # 임차 면적 (supplyArea)
            "${site2}": str(total_floor_area) if total_floor_area else '',  # 건물 면적
            "${site3}": str(area) if area else '',                   # 토지 면적
            "${use}": use_combined,                                  # 건물 구조/용도 (합쳐서)
        })
        
        # 금액 정보
        deposit_price = contract_data.get('depositPrice', 0)
        monthly_rent = contract_data.get('monthlyRent', 0)
        maintenance_fee = contract_data.get('maintenanceFee', 0)
        
        replacements.update({
            "${deposit}": str(deposit_price) if deposit_price else '',
            "${kDeposit}": str(contract_data.get('textDepositPrice', '')),  # 한글 보증금
            "${monthly}": str(monthly_rent) if monthly_rent else '',
            "${maintenance}": str(maintenance_fee) if maintenance_fee else '',
            "${kMaintenance}": str(contract_data.get('textMaintenanceFee', '')),  # 한글 관리비
            "${pd}": str(contract_data.get('paymentDueDay', '')),    # 납부일
            "${ownerAccount}": str(contract_data.get('bankAccount', '')),  # 계좌정보
        })
        
        # 계약 기간 - 입주일
        expected_move_in_year = contract_data.get('expectedMoveInYear', '')
        expected_move_in_month = contract_data.get('expectedMoveInMonth', '')
        expected_move_in_day = contract_data.get('expectedMoveInDay', '')
        
        # 계약 기간 - 퇴거일  
        expected_move_out_year = contract_data.get('expectedMoveOutYear', '')
        expected_move_out_month = contract_data.get('expectedMoveOutMonth', '')
        expected_move_out_day = contract_data.get('expectedMoveOutDay', '')
        
        # 계약일
        contract_date_year = contract_data.get('contractDateYear', '')
        contract_date_month = contract_data.get('contractDateMonth', '')
        contract_date_day = contract_data.get('contractDateDay', '')
        
        replacements.update({
            "${sy}": str(expected_move_in_year) if expected_move_in_year else '',
            "${sm}": str(expected_move_in_month).zfill(2) if expected_move_in_month else '',
            "${sd}": str(expected_move_in_day).zfill(2) if expected_move_in_day else '',
            "${ey}": str(expected_move_out_year) if expected_move_out_year else '',
            "${em}": str(expected_move_out_month).zfill(2) if expected_move_out_month else '',
            "${ed}": str(expected_move_out_day).zfill(2) if expected_move_out_day else '',
            "${nowY}": str(contract_date_year) if contract_date_year else '',
            "${nowM}": str(contract_date_month).zfill(2) if contract_date_month else '',
            "${nowD}": str(contract_date_day).zfill(2) if contract_date_day else '',
        })
        
        # 당사자 정보
        replacements.update({
            "${owner}": str(contract_data.get('ownerNickname', '')),
            "${buyer}": str(contract_data.get('buyerNickname', '')),
            "${ownerAddr}": str(contract_data.get('ownerAddr', '')),
            "${ownerId}": str(contract_data.get('ownerSsn', '')),
            "${ownerPhone}": str(contract_data.get('ownerPhoneNumber', '')),
            "${buyerAddr}": str(contract_data.get('buyerAddr', '')),
            "${buyerId}": str(contract_data.get('buyerSsn', '')),
            "${buyerPhone}": str(contract_data.get('buyerPhoneNumber', '')),
        })
        
        # 조건부 임대인 정보 (미납 세금 있을 때만)
        if has_tax_arrears:
            replacements.update({
                "${owner3}": str(contract_data.get('ownerNickname', '')),  # 미납 세금 있을 때 임대인 이름
            })
        else:
            replacements.update({
                "${owner3}": "",  # 미납 세금 없으면 빈 문자열
            })
        
        # 조건부 임대인 정보 (선순위 확정일자 있을 때만)
        if has_prior_fixed_date:
            replacements.update({
                "${owner5}": str(contract_data.get('ownerNickname', '')),  # 선순위 확정일자 있을 때 임대인 이름
            })
        else:
            replacements.update({
                "${owner5}": "",  # 선순위 확정일자 없으면 빈 문자열
            })
        
        # 특약사항 (리스트로 처리)
        special_terms = contract_data.get('special', [])
        if special_terms:
            replacements["${special}"] = special_terms
        else:
            replacements["${special}"] = [""]
        
        # 서명 플레이스홀더 (조건부 처리)
        replacements.update({
            "${ownerSign1}": "" if has_tax_arrears else "",  # 미납 세금 있을 때만 서명1 (이미지로 대체)
            "${ownerSign2}": "" if has_prior_fixed_date else "",  # 선순위 확정일자 있을 때만 서명2 (이미지로 대체)
            "${ownerSign3}": "",  # 기본 서명3 (이미지로 대체)
            "${buyerSign}": "",  # 임차인 서명 (이미지로 대체)
        })
        
        return replacements
    
    def _number_to_korean(self, number: int) -> str:
        """
        숫자를 한글로 변환
        
        Args:
            number: 변환할 숫자
            
        Returns:
            한글로 변환된 문자열
        """
        if number == 0:
            return "영"
        
        units = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
        big_units = ['', '십', '백', '천']
        bigger_units = ['', '만', '억', '조']
        
        result = []
        big_unit_idx = 0
        
        while number > 0:
            part = number % 10000
            if part > 0:
                part_str = ""
                for i in range(4):
                    digit = part % 10
                    if digit > 0:
                        if i == 0 or digit > 1:
                            part_str = units[digit] + big_units[i] + part_str
                        else:
                            part_str = big_units[i] + part_str
                    part //= 10
                
                if big_unit_idx > 0:
                    part_str += bigger_units[big_unit_idx]
                result.insert(0, part_str)
            
            number //= 10000
            big_unit_idx += 1
        
        return ''.join(result)