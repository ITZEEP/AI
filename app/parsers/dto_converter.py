"""
OCR 결과를 Spring DTO 형식으로 변환하는 유틸리티
"""
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from config.logger_config import get_logger
import re


logger = get_logger(__name__)

class DtoConverter:
    """OCR 결과를 Spring DTO 형식으로 변환하는 클래스"""
    
    @staticmethod
    def convert_building_to_dto(ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """건축물대장 OCR 결과를 Spring BuildingDocumentDto 형식으로 변환"""
        return {
            "siteLocation": ocr_result.get("대지위치", ""),
            "roadAddress": ocr_result.get("도로명주소", ""),
            "totalFloorArea": float(ocr_result.get("연면적", 0)),
            "purpose": DtoConverter._extract_purpose(ocr_result.get("용도")),
            "floorNumber": DtoConverter._extract_floor_number(ocr_result.get("층수")),
            "approvalDate": DtoConverter._format_date(ocr_result.get("사용승인일")),
            "isViolationBuilding": ocr_result.get("위반건축물여부") == "예"
        }
    
    @staticmethod
    def _extract_purpose(purpose_data: Union[str, List[str], None]) -> str:
        """용도 데이터 추출 (리스트인 경우 첫 번째 값 사용)"""
        if not purpose_data:
            return ""
        
        if isinstance(purpose_data, list):
            return purpose_data[0] if purpose_data else ""
        
        return str(purpose_data)
    
    @staticmethod
    def _extract_floor_number(floor_str: Optional[str]) -> int:
        """층수 문자열에서 지상 층수만 추출"""
        if not floor_str:
            return 0
        
        match = re.search(r'지상(\d+)층', floor_str)
        return int(match.group(1)) if match else 0
    
    @staticmethod
    def _format_date(date_str: Optional[str]) -> Optional[str]:
        """날짜 형식을 YYYY-MM-DD로 표준화"""
        if not date_str:
            return None
        
        # YYYY.MM.DD 형식을 YYYY-MM-DD로 변환
        formatted_date = date_str.replace(".", "-")
        
        try:
            datetime.strptime(formatted_date, "%Y-%m-%d")
            return formatted_date
        except ValueError:
            return None
    
    @staticmethod
    def convert_register_to_dto(ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """등기부등본 OCR 결과를 Spring RegistryDocumentDto 형식으로 변환"""
        # 권리 정보 추출
        ownership_info = DtoConverter._extract_ownership_info(ocr_result.get("갑구", []))
        mortgagee_list = DtoConverter._extract_mortgagee_list(ocr_result.get("을구", []))
        address_info = DtoConverter._extract_address_info(ocr_result.get("표제부", {}))
        legal_status = ocr_result.get("법적상태", {})
        
        # 첫 번째 근저당권의 채무자를 debtor 필드로 설정
        debtor_name = ""
        if mortgagee_list and len(mortgagee_list) > 0:
            debtor_name = mortgagee_list[0].get("debtor", "")
        
        return {
            "regionAddress": address_info["region"],
            "roadAddress": address_info["road"],
            "ownerName": ownership_info["owner_name"],
            "ownerBirthDate": None,
            "debtor": debtor_name,
            "mortgageeList": mortgagee_list,
            "hasSeizure": legal_status.get("가압류_여부", False),
            "hasAuction": legal_status.get("경매_여부", False),
            "hasLitigation": legal_status.get("소송_여부", False),
            "hasAttachment": legal_status.get("압류_여부", False)
        }
    
    @staticmethod
    def _extract_ownership_info(gap_gu_data: List[Any]) -> Dict[str, str]:
        """갑구에서 소유자 정보 추출"""
        owner_name = ""
        
        for item in gap_gu_data:
            if isinstance(item, list):
                owner_name = DtoConverter._find_in_dict_list(item, "소유자명")
                if owner_name:
                    break
            elif isinstance(item, str):
                match = re.search(r'소유자\s+([^\s]+)', item)
                if match:
                    owner_name = match.group(1)
                    break
        
        return {"owner_name": owner_name}
    
    @staticmethod
    def _extract_mortgagee_list(eul_gu_data: List[Any]) -> List[Dict[str, Any]]:
        """을구에서 근저당권 목록 추출 (순위번호별로)"""
        mortgagee_list = []
        priority_counter = 1
        
        for item in eul_gu_data:
            if isinstance(item, list):
                # 딕셔너리 리스트에서 정보 추출
                mortgage_info = None
                for data in item:
                    if isinstance(data, dict):
                        # 순위번호 찾기
                        priority_num = None
                        for key, value in data.items():
                            if "순위번호" in str(key):
                                try:
                                    priority_num = int(re.search(r'\d+', str(value)).group()) if re.search(r'\d+', str(value)) else priority_counter
                                except (AttributeError, ValueError) as e:
                                    logger.debug(f"순위번호 추출 실패: {e}")
                                    priority_num = priority_counter
                                break
                        
                        # 근저당권 정보가 있으면 추출
                        if data.get("채권최고액") or data.get("근저당권자"):
                            debtor_value = data.get("채무자", "")
                            mortgage_info = {
                                "priorityNumber": priority_num or priority_counter,
                                "maxClaimAmount": DtoConverter._extract_amount(data.get("채권최고액", "")),
                                "debtor": debtor_value if debtor_value else "미상",
                                "mortgagee": data.get("근저당권자", "")
                            }
                            
                if mortgage_info:
                    mortgagee_list.append(mortgage_info)
                    priority_counter += 1
                    
            elif isinstance(item, str):
                # 문자열에서 정보 추출
                if "근저당권자" in item or "채권최고액" in item:
                    debtor_value = DtoConverter._extract_pattern(item, r'채무자\s+([^\s]+)')
                    mortgage_info = {
                        "priorityNumber": priority_counter,
                        "maxClaimAmount": DtoConverter._extract_amount_from_text(item),
                        "debtor": debtor_value if debtor_value else "미상",
                        "mortgagee": DtoConverter._extract_pattern(item, r'근저당권자\s+([^\s]+)')
                    }
                    if mortgage_info["mortgagee"] or mortgage_info["maxClaimAmount"]:
                        mortgagee_list.append(mortgage_info)
                        priority_counter += 1
        
        return mortgagee_list
    
    @staticmethod
    def _extract_address_info(title_data: Dict[str, Any]) -> Dict[str, str]:
        """표제부에서 주소 정보 추출"""
        address_str = title_data.get("소재지번_건물명칭", "")
        
        if not address_str:
            return {"region": "", "road": ""}
        
        # 괄호로 도로명주소 분리
        match = re.match(r'(.+?)\s*\((.+?)\)', address_str)
        if match:
            return {
                "region": match.group(1).strip(),
                "road": match.group(2).strip()
            }
        
        return {"region": address_str, "road": ""}
    
    @staticmethod
    def _find_in_dict_list(data_list: List[Any], key: str) -> str:
        """딕셔너리 리스트에서 특정 키의 값 찾기"""
        for data in data_list:
            if isinstance(data, dict) and key in data:
                return data[key]
        return ""
    
    @staticmethod
    def _extract_amount(amount_str: str) -> Optional[int]:
        """금액 문자열에서 숫자 추출"""
        if not amount_str:
            return None
        
        match = re.search(r'[\d,]+', amount_str)
        if match:
            try:
                return int(match.group(0).replace(',', ''))
            except ValueError:
                return None
        return None
    
    @staticmethod
    def _extract_amount_from_text(text: str) -> Optional[int]:
        """텍스트에서 채권최고액 추출"""
        match = re.search(r'채권최고액\s*금([\d,]+)원', text)
        if match:
            try:
                return int(match.group(1).replace(',', ''))
            except ValueError:
                return None
        return None
    
    @staticmethod
    def _extract_pattern(text: str, pattern: str) -> str:
        """정규식 패턴으로 텍스트 추출"""
        match = re.search(pattern, text)
        return match.group(1) if match else ""