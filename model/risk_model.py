"""
model/risk_model.py - 행정안전부 도로명주소 API 기반 사기위험도 분석 모델

역할:
1. 우선순위 기반 위험도 판정 (SAFE/WARN/DANGER)
2. 행정안전부 API로 정확한 주소 검증
3. 4개 카테고리별 상세 분석 내용 생성
4. Spring DetailGroup 형태로 정확한 결과 반환
"""

import sys
import os
import re
import requests
from typing import Dict, List, Optional
from datetime import date
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# 프로젝트 루트 경로 설정
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))  # D:\itzip\AI-develop
law_system_path = os.path.join(project_root, "law_system")

# 프로젝트 루트를 sys.path에 추가
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# law_system 경로도 직접 추가
if law_system_path not in sys.path:
    sys.path.insert(0, law_system_path)

# LangChain imports
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as e:
    print(f"ERROR: Failed to import LangChain dependencies: {e}")
    print("Please install with: pip install langchain-core langchain-google-genai")
    raise

load_dotenv()

# 내부 모듈
try:
    from law_system.law_vectorstore import get_law_vectorstore, search_law
    LAW_SYSTEM_AVAILABLE = True
    print("INFO: law_vectorstore 연결 성공")
except ImportError:
    LAW_SYSTEM_AVAILABLE = False
    print("WARNING: law_vectorstore를 사용할 수 없습니다.")

from config.logger_config import get_logger
logger = get_logger(__name__)


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
class DetailAnalysisResult:
    """카테고리별 상세 분석 결과"""
    basic_info_title: str                 # 기본정보 제목
    basic_info_content: str               # 기본정보 내용
    
    rights_info_title: str                # 권리관계 제목
    rights_info_content: str              # 권리관계 내용
    
    building_info_title: str              # 건축관련 제목
    building_info_content: str            # 건축관련 내용
    
    legal_info_title: str                 # 법령위험 제목
    legal_info_content: str               # 법령위험 내용


@dataclass
class RiskAnalysisResult:
    """종합 위험도 분석 결과"""
    risk_level: RiskLevel                 # 종합 위험도
    risk_message: str                     # 위험도 메시지 
    detail_analysis: DetailAnalysisResult # 4개 카테고리 분석
    confidence_score: float = 0.8         # 분석 신뢰도


class JusoApiAddressVerifier:
    """행정안전부 도로명주소 API 기반 주소 검증기 (독립적인 클래스)"""
    
    def __init__(self):
        """행정안전부 도로명주소 API 초기화"""
        self.api_key = os.getenv("JUSO_API_KEY")
        self.base_url = "https://business.juso.go.kr/addrlink/addrLinkApi.do"
        self.cache = {}  # 검증 결과 캐시
        
        if not self.api_key:
            logger.warning("JUSO_API_KEY is not set. Address verification will use fallback method.")
        else:
            # API 키에서 공백 제거
            self.api_key = self.api_key.strip()
            logger.info("JUSO API initialized successfully")
    
    def verify_three_addresses(self, property_addr: str, registry_addr: str, building_addr: str) -> bool:
        """
        3개 주소 동시 검증
        
        Args:
            property_addr: 매물 등록 주소
            registry_addr: 등기부등본 주소  
            building_addr: 건축물대장 주소
            
        Returns:
            bool: 주소 일치 여부
        """
        logger.info("주소 검증 시작:")
        logger.info(f"  매물: {property_addr}")
        logger.info(f"  등기: {registry_addr}")  
        logger.info(f"  건축: {building_addr}")
        
        # API 키가 없으면 자체 검증
        if not self.api_key:
            return self._fallback_verification(property_addr, registry_addr, building_addr)
        
        try:
            # 3개 주소를 모두 API로 정규화
            addresses = [property_addr, registry_addr, building_addr]
            normalized_addresses = []
            
            for addr in addresses:
                if addr and addr.strip():
                    normalized = self._normalize_with_api(addr.strip())
                    normalized_addresses.append(normalized)
                    logger.info(f"  정규화: {addr} → {normalized}")
                else:
                    normalized_addresses.append(None)
            
            # 정규화된 주소들 비교
            valid_addresses = [addr for addr in normalized_addresses if addr]
            
            if len(valid_addresses) < 2:
                logger.warning("유효한 주소가 2개 미만")
                return False
            
            # 정규화된 주소들이 모두 같은 위치인지 확인
            # 도로명주소를 기준으로 비교
            base_addresses = []
            for addr in valid_addresses:
                # 상세 주소 부분 제거 (괄호 안 내용, 동/호수 등)
                base_addr = re.sub(r'\s*\([^)]*\)', '', addr)  # 괄호 제거
                base_addr = re.sub(r'\s*\d+동\s*\d+호.*$', '', base_addr)  # 동호수 제거
                base_addr = base_addr.strip()
                base_addresses.append(base_addr)
            
            unique_addresses = list(set(base_addresses))
            
            if len(unique_addresses) == 1:
                logger.info(f"OK 주소 검증 성공: 모든 주소가 같은 위치 ({unique_addresses[0]})")
                return True
            else:
                logger.warning(f"ERROR 주소 불일치: {len(unique_addresses)}개 서로 다른 위치")
                logger.warning(f"   주소들: {unique_addresses}")
                return False
                
        except Exception as e:
            logger.error(f"API 주소 검증 실패: {e}, 자체 검증으로 전환")
            return self._fallback_verification(property_addr, registry_addr, building_addr)
    
    def _normalize_with_api(self, address: str) -> Optional[str]:
        """행정안전부 API로 주소 정규화"""
        
        # 캐시 확인
        if address in self.cache:
            return self.cache[address]
        
        try:
            # API 요청 파라미터
            params = {
                'confmKey': self.api_key,
                'currentPage': '1',
                'countPerPage': '5',  # 여러 결과 조회
                'keyword': address,
                'resultType': 'json'
            }
            
            # API 호출
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # 응답 파싱
            if data.get('results') and data['results'].get('juso'):
                juso_list = data['results']['juso']
                if juso_list:
                    # 가장 정확한 결과 선택 (도로명주소 우선)
                    best_result = None
                    for result in juso_list:
                        if result.get('roadAddr'):
                            best_result = result
                            break
                    
                    if not best_result and juso_list:
                        best_result = juso_list[0]
                    
                    if best_result:
                        # 도로명주소 우선, 없으면 지번주소
                        normalized = best_result.get('roadAddr') or best_result.get('jibunAddr')
                        
                        # 캐시 저장
                        self.cache[address] = normalized
                        return normalized
            
            # 검색 결과 없음
            logger.warning(f"API에서 주소를 찾을 수 없음: {address}")
            self.cache[address] = None
            return None
            
        except requests.RequestException as e:
            logger.error(f"API 요청 실패 ({address}): {e}")
            return None
        except Exception as e:
            logger.error(f"API 처리 실패 ({address}): {e}")
            return None
    
    def _fallback_verification(self, property_addr: str, registry_addr: str, building_addr: str) -> bool:
        """API 실패시 자체 검증 로직"""
        logger.info("자체 주소 검증 수행")
        
        def extract_key_components(addr: str) -> Dict[str, str]:
            """주소에서 핵심 구성요소 추출"""
            if not addr:
                return {}
            
            components = {}
            
            # 시/도
            sido_match = re.search(r'(서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시|세종특별자치시|경기도|강원특별자치도|충청북도|충청남도|전라북도|전라남도|경상북도|경상남도|제주특별자치도)', addr)
            if sido_match:
                components['sido'] = sido_match.group(1)
            
            # 구/군
            gugun_match = re.search(r'([가-힣]+(?:구|군))', addr)
            if gugun_match:
                components['gugun'] = gugun_match.group(1)
            
            # 동/읍/면
            dong_match = re.search(r'([가-힣]+(?:동|읍|면))', addr)
            if dong_match:
                components['dong'] = dong_match.group(1)
            
            # 로/길
            road_match = re.search(r'([가-힣0-9]+(?:로|길))', addr)
            if road_match:
                components['road'] = road_match.group(1)
            
            return components
        
        # 3개 주소의 핵심 구성요소 추출
        prop_comp = extract_key_components(property_addr)
        reg_comp = extract_key_components(registry_addr)
        build_comp = extract_key_components(building_addr)
        
        # 최소 요구사항: 시도 + 구군이 일치
        def is_same_area(comp1, comp2):
            return (comp1.get('sido') == comp2.get('sido') and 
                   comp1.get('gugun') == comp2.get('gugun'))
        
        # 3개 주소 모두 같은 구/군인지 확인
        same_area_count = 0
        
        if prop_comp and reg_comp and is_same_area(prop_comp, reg_comp):
            same_area_count += 1
        
        if prop_comp and build_comp and is_same_area(prop_comp, build_comp):
            same_area_count += 1
            
        if reg_comp and build_comp and is_same_area(reg_comp, build_comp):
            same_area_count += 1
        
        # 2개 이상 쌍이 같은 지역이면 일치로 판정
        is_match = same_area_count >= 2
        logger.info(f"자체 검증 결과: {is_match} (같은 지역 쌍: {same_area_count}/3)")
        
        return is_match


class RiskAnalysisModel:
    """행정안전부 API 기반 위험도 분석 모델"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.1):
        """
        Args:
            model_name: 사용할 LLM 모델명
            temperature: LLM temperature (일관성을 위해 낮게 설정)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._setup_llm()
        self.vectorstore = self._setup_vectorstore()
        self.address_verifier = JusoApiAddressVerifier()  # OK 독립적인 주소 검증기
        
    def _setup_llm(self):
        """Gemini 1.5 Flash LLM 설정"""
        try:
            # 환경 변수 확인
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
            
            # API 키에서 공백 제거
            api_key = api_key.strip()
            
            # ChatGoogleGenerativeAI 임포트 확인
            if ChatGoogleGenerativeAI is None:
                raise ImportError("ChatGoogleGenerativeAI could not be imported. Please install langchain-google-genai")
            
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=api_key
            )
            logger.info("Success: Gemini LLM initialized.")
            return llm
        except Exception as e:
            logger.error(f"Error: LLM initialization failed: {e}")
            raise
        
    def _setup_vectorstore(self):
        """법령 벡터스토어 설정 - 무조건 로딩"""
        if not LAW_SYSTEM_AVAILABLE:
            raise RuntimeError("law_vectorstore is required but not available! Please check law_system module installation.")
        
        try:
            logger.info("Initializing law vectorstore...")
            vectorstore = get_law_vectorstore()
            if vectorstore:
                logger.info("Success: Vectorstore connected.")
                return vectorstore
            else:
                raise RuntimeError("Vectorstore is None - initialization failed! Check if vectorstore data exists in data/vectorstore/")
        except Exception as e:
            logger.error(f"Error: Vectorstore connection failed: {e}")
            logger.error("Make sure ChromaDB is installed and vectorstore data exists")
            raise
    
    def analyze_risk(self, 
                    user_info: UserInfo,
                    property_info: PropertyInfo,
                    registry_data: RegistryData, 
                    building_data: BuildingData) -> RiskAnalysisResult:
        """
        종합 위험도 분석 수행
        
        Args:
            user_info: 사용자 정보
            property_info: 매물 정보 (Spring DB 데이터)
            registry_data: 등기부등본 데이터 (사용자 검증 완료)
            building_data: 건축물대장 데이터 (사용자 검증 완료)
            
        Returns:
            RiskAnalysisResult: 파싱된 분석 결과
        """
        try:
            logger.info(f"위험도 분석 시작 - user_id: {user_info.user_id}, home_id: {property_info.home_id}")
            
            # OK 1. 주소 검증 먼저 수행 (LLM 프롬프트에서도 사용하기 위함)
            address_match = self.address_verifier.verify_three_addresses(
                property_info.address,
                registry_data.region_address,
                building_data.site_location
            )

            address_summary = (
                "행정안전부 도로명주소 API를 통해 매물 주소, 등기부 주소, 건축물대장 주소는 모두 동일한 위치로 확인되었습니다."
                if address_match else
                "주소 정보는 행정안전부 API 기준 서로 다른 위치로 확인되었습니다. 추가 검토가 필요합니다."
            )
            
            # 1. 우선순위 기반 위험도 판정 (API 주소 검증 포함)
            risk_level = self._determine_risk_level_by_priority(
            user_info, property_info, registry_data, building_data, address_match
            )
            
            # 2. 관련 법령 검색
            relevant_laws = self._search_relevant_laws(registry_data, building_data)
            
            # 3. LLM으로 상세 분석 수행
            detail_analysis = self._analyze_details_with_llm(
                user_info, property_info, registry_data, building_data, relevant_laws, risk_level, address_summary
            )
            
            # 4. 위험도 메시지 생성
            risk_message = self._generate_risk_message(risk_level)
            
            result = RiskAnalysisResult(
                risk_level=risk_level,
                risk_message=risk_message,
                detail_analysis=detail_analysis,
                confidence_score=0.9 if risk_level != RiskLevel.WARN else 0.7
            )
            
            logger.info(f"위험도 분석 완료 - 결과: {result.risk_level}")
            return result
            
        except Exception as e:
            logger.error(f"위험도 분석 실패: {e}")
            return self._get_fallback_result()
    
    def _determine_risk_level_by_priority(self, 
                                        user_info: UserInfo,
                                        property_info: PropertyInfo, 
                                        registry_data: RegistryData, 
                                        building_data: BuildingData,
                                        address_match: bool) -> RiskLevel:
        """우선순위 기반 위험도 판정 (API 주소 검증 포함)"""
        
        # 1순위: 소유자 검증
        if property_info.registered_user_name != registry_data.owner_name:
            logger.warning(f"소유자 불일치 감지: {property_info.registered_user_name} ≠ {registry_data.owner_name}")
            return RiskLevel.DANGER
        
        # 2순위: 근저당권 비율
        mortgage_ratio = self._calculate_mortgage_risk_ratio(registry_data, property_info)
        if mortgage_ratio >= 70:
            logger.warning(f"근저당 비율 위험: {mortgage_ratio:.1f}%")
            return RiskLevel.DANGER
        elif mortgage_ratio > 30:
            logger.info(f"근저당 비율 주의: {mortgage_ratio:.1f}%")
            current_risk = RiskLevel.WARN
        else:
            current_risk = RiskLevel.SAFE
        
        # 3순위: 권리제한 확인
        if any([registry_data.has_seizure, registry_data.has_auction, 
               registry_data.has_litigation, registry_data.has_attachment]):
            logger.warning("권리제한 사항 발견")
            return RiskLevel.DANGER
        
        # 4순위: 주소 일치성
        if not address_match:
            logger.warning("주소 불일치 감지 (API 검증)")
            return RiskLevel.DANGER  # 즉시 반환

        # 5순위: 위반건축물  
        if building_data.is_violation_building:
            logger.warning("위반건축물 감지")
            return RiskLevel.DANGER
        
        return current_risk
    
    def _calculate_mortgage_risk_ratio(self, registry_data: RegistryData, property_info: PropertyInfo) -> float:
        """근저당권 위험 비율 정확한 계산 (모든 근저당권 합산)"""
        if not registry_data.mortgagee_list or not property_info.deposit_price:
            return 0.0
        
        # 모든 근저당권의 채권최고액 합산
        total_max_claim_amount = sum(
            m.max_claim_amount for m in registry_data.mortgagee_list 
            if m.max_claim_amount is not None
        )
        
        if total_max_claim_amount == 0:
            return 0.0
            
        return (total_max_claim_amount / property_info.deposit_price) * 100
    
    def _search_relevant_laws(self, registry_data: RegistryData, building_data: BuildingData) -> List[Dict]:
        """상황별 맞춤 법령 검색"""
        if not self.vectorstore:
            return []
        
        try:
            keywords = []
            
            # 위험 상황별 키워드 추가
            if registry_data.mortgagee_list and len(registry_data.mortgagee_list) > 0:
                keywords.extend(["근저당권", "채권최고액", "우선변제권"])
            
            if any([registry_data.has_seizure, registry_data.has_auction, 
                   registry_data.has_litigation, registry_data.has_attachment]):
                keywords.extend(["가압류", "경매", "소송", "압류"])
            
            if building_data.is_violation_building:
                keywords.extend(["위반건축물", "건축법"])
            
            # 기본 키워드
            keywords.extend(["주택임대차보호법", "전세보증금", "임대차계약"])
            
            search_query = " ".join(list(set(keywords))[:5])
            return search_law(search_query, k=3)
            
        except Exception as e:
            logger.error(f"법령 검색 실패: {e}")
            return []
    
    def _analyze_details_with_llm(self, 
                                 user_info: UserInfo,
                                 property_info: PropertyInfo,
                                 registry_data: RegistryData, 
                                 building_data: BuildingData,
                                 relevant_laws: List[Dict],
                                 risk_level: RiskLevel,
                                 address_summary: str) -> DetailAnalysisResult:
        """LLM을 사용한 4개 카테고리 상세 분석"""
        
        # 상세 분석 프롬프트 템플릿
        detail_analysis_prompt = PromptTemplate(
            input_variables=["user_info", "property_info", "registry_info", "building_info", 
                           "relevant_laws", "risk_level", "mortgage_ratio","address_summary"],
            template="""
당신은 부동산 전문가입니다. 위험도가 '{risk_level}'로 판정된 매물에 대해 4개 카테고리별 상세 분석을 수행해주세요.

## 📊 분석 데이터:
**사용자 정보**: {user_info}
**매물 정보**: {property_info}
**등기부등본**: {registry_info}
**건축물대장**: {building_info}
**근저당 비율**: {mortgage_ratio:.1f}%
**관련 법령**: {relevant_laws}
**주소 정합성**: {address_summary}

## 📋 카테고리별 분석

### 1. 기본정보 분석
**제목**: [소유자 검증 또는 주소 일치성 관련 적절한 제목]
**내용**: 소유자 일치성과 주소 정합성을 중심으로 2~3문장으로 분석하세요.

### 2. 권리관계 분석  
**제목**: [근저당권 또는 권리제한 관련 적절한 제목]
**내용**: 근저당권 비율과 가압류/경매/소송/압류 여부를 중심으로 2~3문장으로 분석하세요.

### 3. 건축관련 분석
**제목**: [건축물 적법성 또는 용도 관련 적절한 제목] 
**내용**: 위반건축물 여부와 매물 타입 일치성을 중심으로 2~3문장으로 분석하세요.

### 4. 법령위험 분석
**제목**: [관련 법령 또는 준수사항 관련 적절한 제목]
**내용**: 관련 법령을 바탕으로 주의사항이나 법적 위험요소를 2~3문장으로 분석하세요.

반드시 위 형식을 정확히 지켜서 응답해주세요.
"""
        )
        
        # 입력 데이터 포맷팅
        user_info_str = self._format_user_info(user_info)
        property_info_str = self._format_property_info(property_info)
        registry_info_str = self._format_registry_data(registry_data)
        building_info_str = self._format_building_data(building_data)
        laws_info_str = self._format_laws_data(relevant_laws)
        mortgage_ratio = self._calculate_mortgage_risk_ratio(registry_data, property_info)
        
        # LLM 호출
        try:
            prompt = detail_analysis_prompt.format(
                user_info=user_info_str,
                property_info=property_info_str,
                registry_info=registry_info_str,
                building_info=building_info_str,
                relevant_laws=laws_info_str,
                risk_level=risk_level.value,
                mortgage_ratio=mortgage_ratio,
                address_summary=address_summary
            )
            
            response = self.llm.invoke(prompt)
            analysis_text = response.content
            
            # 응답 파싱
            return self._parse_detail_analysis_response(analysis_text)
            
        except Exception as e:
            logger.error(f"LLM 상세 분석 실패: {e}")
            return self._get_fallback_detail_analysis()
    
    def _parse_detail_analysis_response(self, response_text: str) -> DetailAnalysisResult:
        """LLM 상세 분석 응답 파싱"""
        try:
            # 기본값 설정
            result = {
                'basic_info_title': '기본 정보 확인',
                'basic_info_content': '분석 결과를 추출할 수 없습니다.',
                'rights_info_title': '권리관계 확인',
                'rights_info_content': '분석 결과를 추출할 수 없습니다.',
                'building_info_title': '건축물 확인',
                'building_info_content': '분석 결과를 추출할 수 없습니다.',
                'legal_info_title': '법령 준수 확인',
                'legal_info_content': '분석 결과를 추출할 수 없습니다.'
            }
            
            # 섹션별 내용 추출
            sections = [
                ('basic_info', r'### 1\. 기본정보 분석(.*?)(?=### 2\.|$)'),
                ('rights_info', r'### 2\. 권리관계 분석(.*?)(?=### 3\.|$)'),
                ('building_info', r'### 3\. 건축관련 분석(.*?)(?=### 4\.|$)'),
                ('legal_info', r'### 4\. 법령위험 분석(.*?)(?=$)')
            ]
            
            for section_name, pattern in sections:
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    section_content = match.group(1).strip()
                    
                    # 제목 추출
                    title_match = re.search(r'\*\*제목\*\*:\s*(.+?)(?:\n|\*\*)', section_content)
                    if title_match:
                        result[f'{section_name}_title'] = title_match.group(1).strip()
                    
                    # 내용 추출
                    content_match = re.search(r'\*\*내용\*\*:\s*(.+?)(?:\n### |$)', section_content, re.DOTALL)
                    if content_match:
                        result[f'{section_name}_content'] = content_match.group(1).strip()
            
            return DetailAnalysisResult(
                basic_info_title=result['basic_info_title'],
                basic_info_content=result['basic_info_content'],
                rights_info_title=result['rights_info_title'],
                rights_info_content=result['rights_info_content'],
                building_info_title=result['building_info_title'],
                building_info_content=result['building_info_content'],
                legal_info_title=result['legal_info_title'],
                legal_info_content=result['legal_info_content']
            )
            
        except Exception as e:
            logger.error(f"상세 분석 응답 파싱 실패: {e}")
            return self._get_fallback_detail_analysis()
    
    def _generate_risk_message(self, risk_level: RiskLevel) -> str:
        """위험도별 메시지 생성"""
        messages = {
            RiskLevel.SAFE: "이 매물은 안전한 상황입니다",
            RiskLevel.WARN: "이 매물은 주의가 필요합니다", 
            RiskLevel.DANGER: "이 매물은 위험 상황입니다"
        }
        return messages.get(risk_level, "분석을 완료했습니다")
    
    def _format_user_info(self, user_info: UserInfo) -> str:
        """사용자 정보 포맷팅"""
        return f"사용자 ID: {user_info.user_id}, 유형: {'임대인' if user_info.user_type == 'landlord' else '임차인'}"
    
    def _format_property_info(self, property_info: PropertyInfo) -> str:
        """매물 정보 포맷팅"""
        return f"""
매물 ID: {property_info.home_id}
주소: {property_info.address}
등록자: {property_info.registered_user_name}
보증금: {f'{property_info.deposit_price:,}원' if property_info.deposit_price else '0원'}
"""
    
    def _format_registry_data(self, data: RegistryData) -> str:
        """등기부등본 데이터 포맷팅"""
        # 근저당권 정보 포맷팅
        mortgage_info = ""
        if data.mortgagee_list and len(data.mortgagee_list) > 0:
            mortgage_lines = []
            total_amount = 0
            for m in data.mortgagee_list:
                amount_str = f'{m.max_claim_amount:,}원' if m.max_claim_amount else '미상'
                if m.max_claim_amount:
                    total_amount += m.max_claim_amount
                mortgage_lines.append(f"  - {m.priority_number}순위: {m.mortgagee} (채권최고액: {amount_str}, 채무자: {m.debtor})")
            mortgage_info = "\n".join(mortgage_lines)
            mortgage_info = f"근저당권 설정:\n{mortgage_info}\n  총 채권최고액: {total_amount:,}원"
        else:
            mortgage_info = "근저당권: 설정없음"
            
        return f"""
소재지번: {data.region_address}
소유자: {data.owner_name}
{mortgage_info}
권리제한: 가압류({data.has_seizure}), 경매({data.has_auction}), 소송({data.has_litigation}), 압류({data.has_attachment})
"""
    
    def _format_building_data(self, data: BuildingData) -> str:
        """건축물대장 데이터 포맷팅"""
        return f"""
대지위치: {data.site_location}
용도: {data.purpose}
연면적: {data.total_floor_area}㎡
위반건축물: {'예' if data.is_violation_building else '아니오'}
"""
    
    def _format_laws_data(self, laws: List[Dict]) -> str:
        """법령 데이터 포맷팅"""
        if not laws:
            return "관련 법령 정보 없음"
        
        formatted = []
        for law in laws:
            law_name = law.get('law_name', '법령명 미상')
            article = law.get('article', '')
            content = law.get('content', '')[:100] + "..." if len(law.get('content', '')) > 100 else law.get('content', '')
            formatted.append(f"- {law_name} {article}: {content}")
        
        return "\n".join(formatted)
    
    def _get_fallback_detail_analysis(self) -> DetailAnalysisResult:
        """오류시 기본 상세 분석"""
        return DetailAnalysisResult(
            basic_info_title="기본 정보 확인",
            basic_info_content="분석 중 오류가 발생했습니다.",
            rights_info_title="권리관계 확인", 
            rights_info_content="분석 중 오류가 발생했습니다.",
            building_info_title="건축물 확인",
            building_info_content="분석 중 오류가 발생했습니다.",
            legal_info_title="법령 준수 확인",
            legal_info_content="분석 중 오류가 발생했습니다."
        )
    
    def _get_fallback_result(self) -> RiskAnalysisResult:
        """오류시 기본 결과"""
        return RiskAnalysisResult(
            risk_level=RiskLevel.WARN,
            risk_message="분석 중 오류가 발생했습니다",
            detail_analysis=self._get_fallback_detail_analysis(),
            confidence_score=0.0
        )


# 사용 예제
if __name__ == "__main__":
    # 환경 변수 확인
    juso_api_key = os.getenv("JUSO_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    print(f"JUSO_API_KEY set: {'Yes' if juso_api_key else 'No'}")
    print(f"GOOGLE_API_KEY set: {'Yes' if google_api_key else 'No'}")
    
    if not juso_api_key:
        print("Warning: JUSO_API_KEY is not set in .env file.")
    if not google_api_key:
        print("Warning: GOOGLE_API_KEY is not set in .env file.")
    
    # 테스트용 데이터
    user_info = UserInfo(user_id=1, user_type="tenant")
    
    property_info = PropertyInfo(
        home_id=1,
        address="서울특별시 광진구 능동로 195-16",
        registered_user_name="홍길동",
        residence_type="APARTMENT",
        lease_type="JEONSE",
        deposit_price=800000000
    )
    
    registry_data = RegistryData(
        region_address="서울특별시 광진구 군자동 98-38",  # 지번주소
        road_address="서울특별시 광진구 능동로 195-16",     # 도로명주소
        owner_name="홍길동",  # 일치
        debtor="홍길동",  # 채무자
        mortgagee_list=[
            MortgageeInfo(
                priority_number=1,
                debtor="홍길동",
                max_claim_amount=2000000000,  # 25% 비율 (안전)
                mortgagee="KB국민은행"
            )
        ],
        has_seizure=False
    )
    
    building_data = BuildingData(
        site_location="서울특별시 광진구 군자동 98-38",
        road_address="서울특별시 광진구 능동로 195-16",
        total_floor_area=84.5,
        purpose="아파트",
        floor_number=15,
        is_violation_building=False
    )
    
    # 분석 실행
    if juso_api_key and google_api_key:
        try:
            model = RiskAnalysisModel()
            result = model.analyze_risk(
                user_info=user_info,
                property_info=property_info,
                registry_data=registry_data,
                building_data=building_data
            )
            
            print("\n=== 위험도 분석 결과 ===")
            print(f"위험도: {result.risk_level}")
            print(f"메시지: {result.risk_message}")
            print(f"신뢰도: {result.confidence_score}")
            print(f"\n기본정보: {result.detail_analysis.basic_info_title}")
            print(f"내용: {result.detail_analysis.basic_info_content}")
            print(f"\n권리관계: {result.detail_analysis.rights_info_title}")
            print(f"내용: {result.detail_analysis.rights_info_content}")
            
        except Exception as e:
            print(f"ERROR 분석 실행 실패: {e}")
    else:
        print("ERROR API 키가 설정되지 않아 분석을 실행할 수 없습니다.")
        
        # 주소 검증기만 테스트
        if juso_api_key:
            print("\n=== 주소 검증 테스트 ===")
            verifier = JusoApiAddressVerifier()
            result = verifier.verify_three_addresses(
                "서울특별시 광진구 능동로 195-16",
                "서울특별시 광진구 군자동 98-38", 
                "서울특별시 광진구 능동로 195-16"
            )
            print(f"주소 일치 결과: {result}")