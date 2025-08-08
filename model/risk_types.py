"""
model/risk_types.py - Risk analysis data types and structures

Shared data types used by both risk_model.py and risk_report.py
to avoid circular imports.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from datetime import date


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