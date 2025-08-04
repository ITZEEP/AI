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
            "siteLocation": f"{ocr_result.get('대지위치', '')} {ocr_result.get('지번', '')}",
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
            return', '.join(purpose_data) if purpose_data else ""
        
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
        # risk_analysis_ready 섹션에서 데이터 추출
        risk_data = ocr_result.get("risk_analysis_ready", {})
        
        if not risk_data:
            raise ValueError("risk_analysis_ready 섹션이 없습니다. 최신 register_parser를 사용하세요.")
        
        return {
            "regionAddress": risk_data.get("region_address", ""),
            "roadAddress": risk_data.get("road_address", ""),
            "ownerName": risk_data.get("owner_name", ""),
            "ownerBirthDate": risk_data.get("owner_birth_date"),
            "debtor": risk_data.get("debtor", ""),
            "mortgageeList": DtoConverter._convert_mortgagee_list(risk_data.get("mortgageeList", [])),
            "hasSeizure": risk_data.get("has_seizure", False),
            "hasAuction": risk_data.get("has_auction", False),
            "hasLitigation": risk_data.get("has_litigation", False),
            "hasAttachment": risk_data.get("has_attachment", False)
        }
    
    @staticmethod
    def _convert_mortgagee_list(mortgagee_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """risk_analysis_ready의 mortgageeList를 Spring DTO 형식으로 변환"""
        converted_list = []
        for item in mortgagee_list:
            converted_item = {
                "priorityNumber": item.get("priorityNumber", 0),
                "maxClaimAmount": item.get("MaxClaimAmount", 0),  # MaxClaimAmount -> maxClaimAmount
                "debtor": item.get("debtor", ""),
                "mortgagee": item.get("mortgagee", "")
            }
            converted_list.append(converted_item)
        return converted_list 