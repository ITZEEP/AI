"""
generators/risk_report.py - Spring 포맷 변환 및 데이터 파싱 (완전한 정적 메서드 버전)

역할:
1. Spring DTO → AI 모델 데이터 변환
2. AI 모델 결과 → Spring DetailGroup 포맷 변환 (카테고리별 위험도 포함)
3. 데이터 파싱 및 검증
"""
import sys
import re
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """위험도 등급"""
    SAFE = "SAFE"      # 안전
    WARN = "WARN"      # 주의
    DANGER = "DANGER"  # 위험


@dataclass
class UserInfo:
    """사용자 정보"""
    user_id: int
    user_type: str  # "landlord" or "tenant"


@dataclass
class PropertyInfo:
    """매물 정보 (Spring DB에서 가져온 실제 등록 정보)"""
    home_id: int
    address: str                          # 등록된 주소 (addr1 + addr2)
    registered_user_name: str             # 매물 등록한 사람 이름
    residence_type: str                   # "APARTMENT", "OFFICETEL" 등
    lease_type: str                       # "JEONSE" or "WOLSE"
    deposit_price: Optional[int] = None   # 보증금
    monthly_rent: Optional[int] = None    # 월세
    maintenance_fee: Optional[int] = None # 관리비


@dataclass
class MortgageeInfo:
    """근저당권 정보"""
    priority_number: int          # 순위번호
    debtor: str                               # 채무자 (필수)
    max_claim_amount: Optional[int] = None    # 채권최고액
    mortgagee: Optional[str] = None           # 근저당권자


@dataclass
class RegistryData:
    """등기부등본 데이터 (Spring에서 사용자 검증 완료된 데이터)"""
    region_address: str                   # 소재지번
    road_address: str                     # 도로명주소
    owner_name: str                       # 소유자명
    owner_birth_date: Optional[date] = None
    debtor: Optional[str] = None              # 채무자 (첫 번째 근저당권의 채무자)
    mortgagee_list: Optional[List[MortgageeInfo]] = None  # 근저당권 목록
    has_seizure: bool = False                 # 가압류 여부
    has_auction: bool = False                 # 경매 여부
    has_litigation: bool = False              # 소송 여부
    has_attachment: bool = False              # 압류 여부


@dataclass
class BuildingData:
    """건축물대장 데이터 (Spring에서 사용자 검증 완료된 데이터)"""
    site_location: str                    # 대지위치
    road_address: str                     # 도로명주소
    total_floor_area: float               # 연면적
    purpose: str                          # 용도
    floor_number: int                     # 층수
    approval_date: Optional[date] = None  # 사용승인일
    is_violation_building: bool = False   # 위반건축물 여부


@dataclass
class CategoryAnalysisResult:
    """개별 카테고리 분석 결과"""
    title: str                            # 카테고리 제목
    content: str                          # 카테고리 내용
    risk_level: RiskLevel                 # 카테고리별 위험도


@dataclass
class DetailAnalysisResult:
    """카테고리별 상세 분석 결과"""
    basic_info: CategoryAnalysisResult    # 기본정보 분석
    rights_info: CategoryAnalysisResult   # 권리관계 분석
    building_info: CategoryAnalysisResult # 건축관련 분석
    legal_info: CategoryAnalysisResult    # 법령위험 분석


@dataclass
class RiskAnalysisResult:
    """종합 위험도 분석 결과"""
    risk_level: RiskLevel                 # 종합 위험도
    risk_message: str                     # 위험도 메시지 
    detail_analysis: DetailAnalysisResult # 4개 카테고리 분석


class RiskReportGenerator:
    """위험도 분석 리포트 생성기 - Spring 연동 전용 (완전한 정적 메서드 버전)"""
    
    @staticmethod
    def generate_spring_risk_report(user_id: int,
                                   user_type: str,
                                   home_id: int,
                                   address: str,
                                   property_price: Optional[int],
                                   lease_type: Optional[str],
                                   spring_registry_dto: Dict[str, Any],
                                   spring_building_dto: Dict[str, Any],
                                   registered_user_name: str, 
                                   residence_type: str,
                                   monthly_rent: Optional[int] = None) -> Dict[str, Any]:
        """
        Spring에서 받은 데이터로 위험도 분석 후 Spring 형태로 반환
        
        Args:
            user_id: 사용자 ID
            user_type: 사용자 타입 ("landlord" or "tenant")
            home_id: 매물 ID
            address: 매물 주소
            property_price: 매물 가격 (전세: 전세금, 월세: 보증금)
            lease_type: 임대 유형 ("JEONSE" or "WOLSE")
            spring_registry_dto: Spring RegistryDocumentDto
            spring_building_dto: Spring BuildingDocumentDto
            registered_user_name: 등록된 사용자 이름
            residence_type: 주거 타입
            monthly_rent: 월세 금액 (월세인 경우에만)
            
        Returns:
            Spring DetailGroup 형태의 분석 결과 (카테고리별 위험도 포함)
        """
        try:
            logger.info(f"Spring 위험도 분석 시작 - user_id: {user_id}, home_id: {home_id}, lease_type: {lease_type}")
        
            # 1. Spring DTO를 AI 모델 데이터로 변환
            user_info = RiskReportGenerator._parse_user_info(user_id, user_type)
            property_info = RiskReportGenerator._parse_property_info(home_id, address, property_price, 
                                                     lease_type, registered_user_name, 
                                                     residence_type, monthly_rent)
            registry_data = RiskReportGenerator._parse_spring_registry_dto(spring_registry_dto)
            building_data = RiskReportGenerator._parse_spring_building_dto(spring_building_dto)
            
            # 2. AI 모델로 위험도 분석 (카테고리별 분석 포함)
            from model.risk_model import RiskAnalysisModel
            risk_model = RiskAnalysisModel()
            analysis_result = risk_model.analyze_risk_with_categories(
                user_info=user_info,
                property_info=property_info,
                registry_data=registry_data,
                building_data=building_data
            )
            
            # 3. Spring DetailGroup 형태로 변환 (카테고리별 위험도 포함)
            spring_response = RiskReportGenerator._convert_to_spring_format_with_categories(analysis_result)
            
            logger.info(f"Spring 위험도 분석 완료 - 결과: {analysis_result.risk_level}")
            return spring_response
        
        except Exception as e:
            logger.error(f"Spring 위험도 분석 실패: {e}")
            return RiskReportGenerator._get_spring_fallback_response()
    
    @staticmethod
    def _parse_user_info(user_id: int, user_type: str) -> UserInfo:
        """사용자 정보 파싱"""
        return UserInfo(
            user_id=user_id,
            user_type=user_type
        )
    
    @staticmethod
    def _parse_property_info(home_id: int, address: str, 
                           property_price: Optional[int], lease_type: Optional[str],
                           registered_user_name: str, residence_type: str,
                           monthly_rent: Optional[int] = None) -> PropertyInfo:
        """매물 정보 파싱"""
        
        deposit_price = None
        monthly_rent_amount = None
        
        if lease_type == "JEONSE":
            # 전세인 경우: property_price가 전세금
            deposit_price = property_price
            monthly_rent_amount = None
        elif lease_type == "WOLSE":
            # 월세인 경우: property_price가 보증금, monthly_rent가 월세
            deposit_price = property_price
            monthly_rent_amount = monthly_rent

        return PropertyInfo(
            home_id=home_id,
            address=address,
            registered_user_name=registered_user_name,
            residence_type=residence_type,
            lease_type=lease_type,
            deposit_price=deposit_price,
            monthly_rent=monthly_rent_amount
        )
    
    @staticmethod
    def _parse_spring_registry_dto(spring_dto: Dict[str, Any]) -> RegistryData:
        """Spring RegistryDocumentDto → RegistryData 변환"""
        try:
            # 생년월일 파싱
            birth_date = None
            if spring_dto.get('ownerBirthDate'):
                try:
                    birth_date = datetime.strptime(spring_dto['ownerBirthDate'], '%Y-%m-%d').date()
                except (ValueError, TypeError):
                    logger.warning(f"생년월일 파싱 실패: {spring_dto.get('ownerBirthDate')}")
            
            # 근저당권 목록 파싱
            mortgagee_list = []
            if spring_dto.get('mortgageeList'):
                for m in spring_dto.get('mortgageeList', []):
                    try:
                        debtor_value = m.get('debtor', '')
                        mortgagee_info = MortgageeInfo(
                            priority_number=m.get('priorityNumber', 1),
                            debtor=debtor_value if debtor_value else "미상",
                            max_claim_amount=m.get('maxClaimAmount'),
                            mortgagee=m.get('mortgagee', '')
                        )
                        mortgagee_list.append(mortgagee_info)
                    except Exception as e:
                        logger.warning(f"근저당권 정보 파싱 실패: {e}")
            
            return RegistryData(
                region_address=spring_dto.get('regionAddress', ''),
                road_address=spring_dto.get('roadAddress', ''),
                owner_name=spring_dto.get('ownerName', ''),
                owner_birth_date=birth_date,
                debtor=spring_dto.get('debtor', ''),
                mortgagee_list=mortgagee_list if mortgagee_list else None,
                has_seizure=spring_dto.get('hasSeizure', False),
                has_auction=spring_dto.get('hasAuction', False),
                has_litigation=spring_dto.get('hasLitigation', False),
                has_attachment=spring_dto.get('hasAttachment', False)
            )
            
        except Exception as e:
            logger.error(f"등기부등본 DTO 파싱 실패: {e}")
            # 기본값으로 생성
            return RegistryData(
                region_address="파싱 실패",
                road_address="파싱 실패",
                owner_name="파싱 실패",
                debtor=""
            )
    
    @staticmethod
    def _parse_spring_building_dto(spring_dto: Dict[str, Any]) -> BuildingData:
        """Spring BuildingDocumentDto → BuildingData 변환"""
        try:
            # 사용승인일 파싱
            approval_date = None
            if spring_dto.get('approvalDate'):
                try:
                    approval_date = datetime.strptime(spring_dto['approvalDate'], '%Y-%m-%d').date()
                except (ValueError, TypeError):
                    logger.warning(f"사용승인일 파싱 실패: {spring_dto.get('approvalDate')}")
            
            # 연면적 파싱 (안전한 float 변환)
            total_floor_area = 0.0
            try:
                total_floor_area = float(spring_dto.get('totalFloorArea', 0))
            except (ValueError, TypeError):
                logger.warning(f"연면적 파싱 실패: {spring_dto.get('totalFloorArea')}")
            
            # 층수 파싱 (안전한 int 변환)
            floor_number = 0
            try:
                floor_number = int(spring_dto.get('floorNumber', 0))
            except (ValueError, TypeError):
                logger.warning(f"층수 파싱 실패: {spring_dto.get('floorNumber')}")
            
            return BuildingData(
                site_location=spring_dto.get('siteLocation', ''),
                road_address=spring_dto.get('roadAddress', ''),
                total_floor_area=total_floor_area,
                purpose=spring_dto.get('purpose', ''),
                floor_number=floor_number,
                approval_date=approval_date,
                is_violation_building=spring_dto.get('isViolationBuilding', False)
            )
            
        except Exception as e:
            logger.error(f"건축물대장 DTO 파싱 실패: {e}")
            # 기본값으로 생성
            return BuildingData(
                site_location="파싱 실패",
                road_address="파싱 실패",
                total_floor_area=0.0,
                purpose="파싱 실패",
                floor_number=0
            )
    
    @staticmethod
    def _convert_to_spring_format_with_categories(analysis_result: RiskAnalysisResult) -> Dict[str, Any]:
        """AI 분석 결과를 Spring DetailGroup 형태로 변환 (아이템별 위험도만 포함)"""
        
        # Spring DetailGroup 구조 생성 (아이템별 위험도만 포함)
        detail_groups = [
            {
                "title": "기본 정보",
                "items": [
                    {
                        "title": analysis_result.detail_analysis.basic_info.title,
                        "content": analysis_result.detail_analysis.basic_info.content,
                        "riskLevel": analysis_result.detail_analysis.basic_info.risk_level.value
                    }
                ]
            },
            {
                "title": "권리관계 정보",
                "items": [
                    {
                        "title": analysis_result.detail_analysis.rights_info.title,
                        "content": analysis_result.detail_analysis.rights_info.content,
                        "riskLevel": analysis_result.detail_analysis.rights_info.risk_level.value
                    }
                ]
            },
            {
                "title": "건축 관련",
                "items": [
                    {
                        "title": analysis_result.detail_analysis.building_info.title,
                        "content": analysis_result.detail_analysis.building_info.content,
                        "riskLevel": analysis_result.detail_analysis.building_info.risk_level.value
                    }
                ]
            },
            {
                "title": "법령 위험",
                "items": [
                    {
                        "title": analysis_result.detail_analysis.legal_info.title,
                        "content": analysis_result.detail_analysis.legal_info.content,
                        "riskLevel": analysis_result.detail_analysis.legal_info.risk_level.value
                    }
                ]
            }
        ]
        
        return {
            "riskType": analysis_result.risk_level.value,  # "SAFE", "WARN", "DANGER"
            "riskMessage": analysis_result.risk_message,   # "이 매물은 위험 상황입니다"
            "analyzedAt": datetime.now().isoformat(),
            "detailGroups": detail_groups
        }
        
    @staticmethod
    def _get_spring_fallback_response() -> Dict[str, Any]:
        """오류시 Spring 기본 응답"""
        return {
            "riskType": "WARN",
            "riskMessage": "분석 중 오류가 발생했습니다",
            "analyzedAt": datetime.now().isoformat(),
            "detailGroups": [
                {
                    "title": "기본 정보",
                    "items": [
                        {
                            "title": "시스템 오류",
                            "content": "기본 정보 분석 중 오류가 발생했습니다.",
                            "riskLevel": "WARN"
                        }
                    ]
                },
                {
                    "title": "권리관계 정보",
                    "items": [
                        {
                            "title": "시스템 오류", 
                            "content": "권리관계 분석 중 오류가 발생했습니다.",
                            "riskLevel": "WARN"
                        }
                    ]
                },
                {
                    "title": "건축 관련",
                    "items": [
                        {
                            "title": "시스템 오류",
                            "content": "건축물 분석 중 오류가 발생했습니다.",
                            "riskLevel": "WARN"
                        }
                    ]
                },
                {
                    "title": "법령 위험",
                    "items": [
                        {
                            "title": "시스템 오류",
                            "content": "법령 분석 중 오류가 발생했습니다. 전문가 상담을 권장합니다.",
                            "riskLevel": "WARN"
                        }
                    ]
                }
            ]
        }


class OCRDataParser:
    """OCR 데이터 파싱 유틸리티 클래스"""
    
    @staticmethod
    def parse_registry_ocr_to_spring_dto(ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """등기부등본 OCR 결과를 Spring RegistryDocumentDto 형태로 변환"""
        try:
            # OCR 데이터 구조 분석
            gabgu = ocr_data.get("갑구", [])
            eulgu = ocr_data.get("을구", [])
            title = ocr_data.get("표제부", {})
            legal_status = ocr_data.get("법적상태", {})
            
            # Spring DTO 구조 생성
            registry_dto = {
                "regionAddress": title.get("소재지번_건물명칭", ""),
                "roadAddress": "",  # OCR에서 도로명주소는 별도 처리 필요
                "ownerName": "",
                "ownerBirthDate": None,
                "maxClaimAmount": None,
                "debtor": None,
                "mortgagee": None,
                "hasSeizure": legal_status.get("가압류_여부", False),
                "hasAuction": legal_status.get("경매_여부", False),
                "hasLitigation": legal_status.get("소송_여부", False),
                "hasAttachment": legal_status.get("압류_여부", False)
            }
            
            # 갑구에서 소유자 정보 추출
            owner_name = OCRDataParser._extract_owner_from_gabgu(gabgu)
            if owner_name:
                registry_dto["ownerName"] = owner_name
            
            # 을구에서 근저당 정보 추출
            mortgage_info = OCRDataParser._extract_mortgage_from_eulgu(eulgu)
            if mortgage_info:
                registry_dto.update(mortgage_info)
            
            return registry_dto
            
        except Exception as e:
            logger.error(f"등기부등본 OCR 파싱 실패: {e}")
            return OCRDataParser._get_default_registry_dto()
    
    @staticmethod
    def parse_building_ocr_to_spring_dto(ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """건축물대장 OCR 결과를 Spring BuildingDocumentDto 형태로 변환"""
        try:
            building_dto = {
                "siteLocation": ocr_data.get("대지위치", ""),
                "roadAddress": ocr_data.get("도로명주소", ""),
                "totalFloorArea": OCRDataParser._safe_float(ocr_data.get("연면적", 0)),
                "purpose": OCRDataParser._format_purpose(ocr_data.get("용도", [])),
                "floorNumber": OCRDataParser._safe_int(ocr_data.get("층수", 0)),
                "approvalDate": OCRDataParser._format_date(ocr_data.get("사용승인일")),
                "isViolationBuilding": ocr_data.get("위반건축물여부") == "예"
            }
            
            return building_dto
            
        except Exception as e:
            logger.error(f"건축물대장 OCR 파싱 실패: {e}")
            return OCRDataParser._get_default_building_dto()
    
    @staticmethod
    def _extract_owner_from_gabgu(gabgu_data) -> Optional[str]:
        """갑구 데이터에서 소유자명 추출"""
        if not gabgu_data:
            return None
        
        try:
            # 다양한 갑구 데이터 구조에 대응
            for item in gabgu_data:
                if isinstance(item, list):
                    for detail in item:
                        if isinstance(detail, dict) and "소유자명" in detail:
                            return detail["소유자명"]
                elif isinstance(item, dict) and "소유자명" in item:
                    return item["소유자명"]
                elif isinstance(item, str) and "소유자" in item:
                    # 텍스트에서 소유자명 추출
                    match = re.search(r'소유자\s+([^\s]+)', item)
                    if match:
                        return match.group(1)
        except Exception as e:
            logger.error(f"소유자명 추출 실패: {e}")
        
        return None
    
    @staticmethod
    def _extract_mortgage_from_eulgu(eulgu_data) -> Dict[str, Any]:
        """을구 데이터에서 근저당 정보 추출"""
        result = {
            "maxClaimAmount": None,
            "debtor": None,
            "mortgagee": None
        }
        
        if not eulgu_data:
            return result
        
        try:
            for item in eulgu_data:
                if isinstance(item, list):
                    for detail in item:
                        if isinstance(detail, dict):
                            if "채권최고액" in detail:
                                result["maxClaimAmount"] = OCRDataParser._extract_amount_from_korean(detail["채권최고액"])
                            if "채무자" in detail:
                                result["debtor"] = detail["채무자"]
                            if "근저당권자" in detail:
                                result["mortgagee"] = detail["근저당권자"]
        except Exception as e:
            logger.error(f"근저당 정보 추출 실패: {e}")
        
        return result
    
    @staticmethod
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
    
    @staticmethod
    def _safe_float(value) -> float:
        """안전한 float 변환"""
        try:
            return float(value) if value else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    @staticmethod
    def _safe_int(value) -> int:
        """안전한 int 변환"""
        try:
            if isinstance(value, str):
                # "지하1층/지상15층" 같은 형태에서 지상 층수 추출
                above_ground = re.search(r'지상(\d+)층', value)
                if above_ground:
                    return int(above_ground.group(1))
                # 일반 숫자 추출
                numbers = re.findall(r'\d+', value)
                if numbers:
                    return int(numbers[-1])
            return int(value) if value else 0
        except (ValueError, TypeError):
            return 0
    
    @staticmethod
    def _format_purpose(purpose_data) -> str:
        """용도 데이터 포맷팅"""
        try:
            if isinstance(purpose_data, list):
                return ", ".join(purpose_data)
            elif isinstance(purpose_data, str):
                return purpose_data
        except Exception:
            pass
        return ""
    
    @staticmethod
    def _format_date(date_str) -> Optional[str]:
        """날짜 포맷팅 (yyyy-MM-dd 형태로 변환)"""
        if not date_str:
            return None
        
        try:
            
            # "2020.03.20" 형태
            if re.match(r'\d{4}\.\d{2}\.\d{2}', date_str):
                return date_str.replace('.', '-')
            
            # "2020년 3월 20일" 형태
            year_match = re.search(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일?', date_str)
            if year_match:
                year, month, day = year_match.groups()
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            # 기타 형태는 그대로 반환
            return date_str
            
        except Exception as e:
            logger.error(f"날짜 포맷팅 실패: {e}")
            return None
    
    @staticmethod
    def _get_default_registry_dto() -> Dict[str, Any]:
        """기본 등기부등본 DTO"""
        return {
            "regionAddress": "OCR 처리 실패",
            "roadAddress": "",
            "ownerName": "OCR 처리 실패",
            "ownerBirthDate": None,
            "maxClaimAmount": None,
            "debtor": None,
            "mortgagee": None,
            "hasSeizure": False,
            "hasAuction": False,
            "hasLitigation": False,
            "hasAttachment": False
        }
    
    @staticmethod
    def _get_default_building_dto() -> Dict[str, Any]:
        """기본 건축물대장 DTO"""
        return {
            "siteLocation": "OCR 처리 실패",
            "roadAddress": "",
            "totalFloorArea": 0.0,
            "purpose": "OCR 처리 실패",
            "floorNumber": 0,
            "approvalDate": None,
            "isViolationBuilding": False
        }


# 편의 함수들
def generate_risk_report_for_spring(user_id: int,
                                   user_type: str,
                                   home_id: int,
                                   address: str,
                                   property_price: Optional[int],
                                   lease_type: Optional[str],
                                   spring_registry_dto: Dict[str, Any],
                                   spring_building_dto: Dict[str, Any],
                                   registered_user_name: str,
                                   residence_type: str,
                                   monthly_rent: Optional[int] = None) -> Dict[str, Any]:
    """Spring용 위험도 분석 리포트 생성 편의 함수"""
    return RiskReportGenerator.generate_spring_risk_report(
        user_id=user_id,
        user_type=user_type,
        home_id=home_id,
        address=address,
        property_price=property_price,
        lease_type=lease_type,
        spring_registry_dto=spring_registry_dto,
        spring_building_dto=spring_building_dto,
        registered_user_name=registered_user_name,
        residence_type=residence_type,
        monthly_rent=monthly_rent
    )


def parse_ocr_data_for_spring(registry_ocr_data: Dict[str, Any], 
                             building_ocr_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """OCR 데이터를 Spring DTO 형태로 파싱하는 편의 함수"""
    registry_dto = OCRDataParser.parse_registry_ocr_to_spring_dto(registry_ocr_data)
    building_dto = OCRDataParser.parse_building_ocr_to_spring_dto(building_ocr_data)
    return registry_dto, building_dto