"""
model/risk_model.py - 행정안전부 도로명주소 API 기반 사기위험도 분석 모델

역할:
1. 우선순위 기반 위험도 판정 (SAFE/WARN/DANGER)
2. 행정안전부 API로 정확한 주소 검증
3. 4개 카테고리별 개별 위험도 분석 및 상세 분석 내용 생성
4. Spring DetailGroup 형태로 정확한 결과 반환
"""

import sys
import os
import re
import requests
from typing import Dict, List, Optional
from datetime import date
from enum import Enum
from dotenv import load_dotenv



# 프로젝트 루트 경로 설정
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
law_system_path = os.path.join(project_root, "law_system")

# 프로젝트 루트를 sys.path에 추가
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# law_system 경로도 직접 추가
if law_system_path not in sys.path:
    sys.path.insert(0, law_system_path)
    
# Import shared data types from risk_types to avoid circular import
from model.risk_types import RiskAnalysisResult, CategoryAnalysisResult, DetailAnalysisResult, RiskLevel

# LangChain imports
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers import StrOutputParser
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
from config.gemini_retry import retry_gemini_api


class JusoApiAddressVerifier:
    """행정안전부 도로명주소 API 기반 주소 검증기"""
    
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
    
    def __init__(self, model_name: str = "gemini-2.5-pro", temperature: float = 0.1):
        """
        Args:
            model_name: 사용할 LLM 모델명
            temperature: LLM temperature (일관성을 위해 낮게 설정)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._setup_llm()
        self.vectorstore = self._setup_vectorstore()
        self.address_verifier = JusoApiAddressVerifier()
        
    def _setup_llm(self):
        """Gemini LLM 설정"""
        try:
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
    
    @retry_gemini_api(max_retries=5, initial_delay=2.0, backoff_multiplier=1.5)
    def _call_gemini_api_for_risk(self, chain, invoke_params):
        """
        Gemini API 호출 래퍼 메서드 (위험도 분석용, 재시도 로직 적용)
        
        Args:
            chain: LangChain 체인
            invoke_params: invoke에 전달할 파라미터
        
        Returns:
            API 호출 결과
        """
        logger.debug("Gemini API 호출 시작 (위험도 분석)")
        result = chain.invoke(invoke_params)
        logger.debug("Gemini API 호출 성공 (위험도 분석)")
        return result
    
    def analyze_risk_with_categories(self, user_info, property_info, registry_data, building_data):
        """
        카테고리별 개별 위험도 분석을 포함한 종합 위험도 분석 수행
        
        Returns:
            RiskAnalysisResult: 카테고리별 위험도가 포함된 분석 결과
        """
        try:
            logger.info(f"카테고리별 위험도 분석 시작 - user_id: {user_info.user_id}, home_id: {property_info.home_id}")
            
            # 1. 주소 검증 먼저 수행
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
            
            # 2. 카테고리별 개별 위험도 분석 (한 번만 수행)
            category_risks = self._analyze_category_risks(
                user_info, property_info, registry_data, building_data, address_match
            )
            
            # 3. 카테고리별 위험도를 바탕으로 종합 위험도 결정 (로직 중복 제거)
            overall_risk_level = self._determine_overall_risk_from_categories(category_risks)
            
            # 4. 관련 법령 검색
            relevant_laws = self._search_relevant_laws(registry_data, building_data)
            
            # 5. LLM으로 카테고리별 상세 분석 수행
            detail_analysis = self._analyze_details_with_llm_by_category(
                user_info, property_info, registry_data, building_data, 
                relevant_laws, category_risks, address_summary
            )
            
            # 6. 위험도 메시지 생성
            risk_message = self._generate_risk_message(overall_risk_level)
            
            
            result = RiskAnalysisResult(
                risk_level=overall_risk_level,
                risk_message=risk_message,
                detail_analysis=detail_analysis
            )
            
            logger.info(f"카테고리별 위험도 분석 완료 - 종합결과: {result.risk_level}")
            logger.info(f"카테고리별 결과: {category_risks}")
            return result
            
        except Exception as e:
            logger.error(f"카테고리별 위험도 분석 실패: {e}")
            return self._get_fallback_result()
    
    def _analyze_category_risks(self, user_info, property_info, registry_data, building_data, address_match) -> Dict[str, RiskLevel]:
        """
        각 카테고리별 개별 위험도 분석 (통합된 단일 분석 로직)
        
        Returns:
            Dict[str, RiskLevel]: 카테고리별 위험도 매핑
        """
        category_risks = {}
        
        # 1. 기본정보 카테고리 위험도 (소유자 검증 + 주소 일치성)
        logger.info("=== 기본정보 카테고리 분석 ===")
        basic_risk = RiskLevel.SAFE
        
        if property_info.registered_user_name != registry_data.owner_name:
            logger.warning(f"소유자 불일치: {property_info.registered_user_name} ≠ {registry_data.owner_name}")
            basic_risk = RiskLevel.DANGER
        elif not address_match:
            logger.warning("주소 불일치 (API 검증)")
            basic_risk = RiskLevel.DANGER
        else:
            logger.info("소유자 일치 및 주소 검증 완료")
            
        category_risks['basic_info'] = basic_risk
        
        # 2. 권리관계 카테고리 위험도 (근저당권 + 권리제한)
        logger.info("=== 권리관계 카테고리 분석 ===")
        rights_risk = RiskLevel.SAFE
        
        # 근저당권 비율 계산
        mortgage_ratio = self._calculate_mortgage_risk_ratio(registry_data, property_info)
        logger.info(f"근저당권 비율: {mortgage_ratio:.1f}%")
        
        if mortgage_ratio >= 70:
            logger.warning(f"근저당 비율 위험: {mortgage_ratio:.1f}%")
            rights_risk = RiskLevel.DANGER
        elif mortgage_ratio > 30:
            logger.info(f"근저당 비율 주의: {mortgage_ratio:.1f}%")
            rights_risk = RiskLevel.WARN
        
        # 권리제한 확인 (더 높은 우선순위)
        legal_restrictions = [
            ("가압류", registry_data.has_seizure),
            ("경매", registry_data.has_auction), 
            ("소송", registry_data.has_litigation),
            ("압류", registry_data.has_attachment)
        ]
        
        active_restrictions = [name for name, status in legal_restrictions if status]
        if active_restrictions:
            logger.warning(f"권리제한 사항 발견: {', '.join(active_restrictions)}")
            rights_risk = RiskLevel.DANGER  # 권리제한이 있으면 무조건 위험
        
        category_risks['rights_info'] = rights_risk
        
        # 3. 건축관련 카테고리 위험도 (위반건축물 + 용도 일치성)
        logger.info("=== 건축관련 카테고리 분석 ===")
        building_risk = RiskLevel.SAFE
        
        if building_data.is_violation_building:
            logger.warning("위반건축물 감지")
            building_risk = RiskLevel.DANGER
        elif not self._check_building_purpose_match(building_data.purpose, property_info.residence_type):
            logger.info(f"용도 불일치: 건축물({building_data.purpose}) vs 매물타입({property_info.residence_type})")
            building_risk = RiskLevel.WARN
        else:
            logger.info("건축물 적법성 및 용도 일치 확인")
            
        category_risks['building_info'] = building_risk
        
        # 4. 법령위험 카테고리 위험도 (전세사기 패턴 + 기타 법적 위험)
        logger.info("=== 법령위험 카테고리 분석 ===")
        legal_risk = RiskLevel.SAFE
        
        # 전세사기 고위험 패턴 체크
        if self._check_jeonse_fraud_pattern(registry_data, property_info):
            logger.warning("전세사기 고위험 패턴 감지")
            legal_risk = RiskLevel.WARN
        else:
            logger.info("전세사기 위험 패턴 없음")
        
        category_risks['legal_info'] = legal_risk
        
        return category_risks
    
    def _determine_overall_risk_from_categories(self, category_risks: Dict[str, RiskLevel]) -> RiskLevel:
        """
        카테고리별 위험도를 바탕으로 종합 위험도 결정 (논리적 흐름)
        
        Args:
            category_risks: 카테고리별 위험도 딕셔너리
            
        Returns:
            RiskLevel: 종합 위험도
        """
        # 위험도 우선순위: DANGER > WARN > SAFE
        risk_counts = {
            RiskLevel.DANGER: 0,
            RiskLevel.WARN: 0,
            RiskLevel.SAFE: 0
        }
        
        # 각 카테고리별 위험도 카운트
        for category, risk_level in category_risks.items():
            risk_counts[risk_level] += 1
            logger.info(f"카테고리 '{category}': {risk_level.value}")
        
        # 종합 위험도 결정 로직
        if risk_counts[RiskLevel.DANGER] > 0:
            logger.warning(f"DANGER 카테고리 {risk_counts[RiskLevel.DANGER]}개 발견 → 종합 위험도: DANGER")
            return RiskLevel.DANGER
        elif risk_counts[RiskLevel.WARN] > 0:
            logger.info(f"WARN 카테고리 {risk_counts[RiskLevel.WARN]}개 발견 → 종합 위험도: WARN")
            return RiskLevel.WARN
        else:
            logger.info("모든 카테고리 안전 → 종합 위험도: SAFE")
            return RiskLevel.SAFE
    
    def _calculate_mortgage_risk_ratio(self, registry_data, property_info) -> float:
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
    
    def _check_building_purpose_match(self, building_purpose: str, residence_type: str) -> bool:
        """건축물 용도와 매물 타입 일치성 확인"""
        purpose_mapping = {
            "APARTMENT": ["아파트", "공동주택"],
            "OFFICETEL": ["오피스텔", "업무시설"],
            "VILLA": ["다세대주택", "연립주택"],
            "ONE_ROOM": ["원룸", "단독주택", "다가구주택"]
        }
        
        expected_purposes = purpose_mapping.get(residence_type, [])
        return any(purpose in building_purpose for purpose in expected_purposes)
    
    def _check_jeonse_fraud_pattern(self, registry_data, property_info) -> bool:
        """전세사기 고위험 패턴 체크"""
        # 전세인데 근저당권이 많거나, 소유자와 채무자가 다른 경우 등
        if property_info.lease_type == "JEONSE":
            if registry_data.mortgagee_list and len(registry_data.mortgagee_list) > 2:
                return True
            
            if (registry_data.debtor and registry_data.owner_name and 
                registry_data.debtor != registry_data.owner_name):
                return True
        
        return False
    
    def _search_relevant_laws(self, registry_data, building_data) -> List[Dict]:
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
    
    def _analyze_details_with_llm_by_category(self, user_info, property_info, registry_data, building_data, 
                                        relevant_laws, category_risks, address_summary):
        """LLM을 사용한 카테고리별 상세 분석 (개별 위험도 포함)"""
        
        # 카테고리별 개별 분석 프롬프트 템플릿
        category_analysis_prompt = PromptTemplate(
        input_variables=["user_info", "property_info", "registry_info", "building_info", 
                       "relevant_laws", "category_risks", "mortgage_ratio", "address_summary"],
            template="""
당신은 부동산 전문가입니다. 4개 카테고리별로 개별 위험도가 판정된 매물에 대해 상세 분석을 수행해주세요.

## 분석 데이터:
**사용자 정보**: {user_info}
**매물 정보**: {property_info}
**등기부등본**: {registry_info}
**건축물대장**: {building_info}
**근저당 비율**: {mortgage_ratio:.1f}%
**관련 법령**: {relevant_laws}
**주소 정합성**: {address_summary}
**카테고리별 위험도**: {category_risks}

## 카테고리별 분석

### 1. 기본정보 분석 
제목: [소유자 검증 또는 주소 일치성 관련 적절한 제목]
내용: 소유자 일치성과 주소 정합성을 중심으로 2~3문장으로 분석하세요.

### 2. 권리관계 분석
제목: [근저당권 또는 권리제한 관련 적절한 제목]
내용: 근저당권 비율과 가압류/경매/소송/압류 여부를 중심으로 2~3문장으로 분석하세요.

### 3. 건축관련 분석 
제목: [건축물 적법성 또는 용도 관련 적절한 제목]
내용: 위반건축물 여부와 매물 타입 일치성을 중심으로 2~3문장으로 분석하세요.

### 4. 법령위험 분석 
제목: [관련 법령 또는 준수사항 관련 적절한 제목]
내용: 관련 법령을 바탕으로 주의사항이나 법적 위험요소를 2~3문장으로 분석하세요.

반드시 위 형식을 정확히 지켜서 응답해주세요. 
중요: **볼드**, *이탤릭* 등 마크다운 문법을 사용하지 마세요. 순수 텍스트로만 작성해주세요.
"""
        )
        
        # LLM 체인 구성
        chain = category_analysis_prompt | self.llm | StrOutputParser()
        
        # 입력 데이터 포맷팅
        user_info_str = self._format_user_info(user_info)
        property_info_str = self._format_property_info(property_info)
        registry_info_str = self._format_registry_data(registry_data)
        building_info_str = self._format_building_data(building_data)
        laws_info_str = self._format_laws_data(relevant_laws)
        mortgage_ratio = self._calculate_mortgage_risk_ratio(registry_data, property_info)
        
        # LLM 호출
        for attempt in range(3):
            try:
                logger.debug(f"위험도 분석 시도 {attempt + 1}/3")
                
                # 재시도 로직이 적용된 API 호출 (gemini_retry로 5번 재시도)
                result = self._call_gemini_api_for_risk(chain, {
                    "user_info": user_info_str,
                    "property_info": property_info_str,
                    "registry_info": registry_info_str,
                    "building_info": building_info_str,
                    "relevant_laws": laws_info_str,
                    "category_risks": {k: (v.value if hasattr(v, "value") else str(v)) for k, v in category_risks.items()},
                    "mortgage_ratio": mortgage_ratio,
                    "address_summary": address_summary
                })
                
                logger.debug(f"위험도 분석 API 완료 - 시도 {attempt + 1}")
                
                # 응답 파싱 후 카테고리별 위험도 추가 (result는 이미 문자열)
                detail_analysis = self._parse_detail_analysis_response_with_categories(result, category_risks)
                
                # 4개 카테고리가 모두 파싱되면 성공
                if self._is_valid_parsed_result(detail_analysis):
                    logger.info(f"위험도 분석 성공 - 시도 {attempt + 1}")
                    return detail_analysis
                else:
                    logger.warning(f"위험도 분석 파싱 실패 - 시도 {attempt + 1}, 재시도 진행")
                    if attempt < 2:
                        continue
                
            except Exception as e:
                logger.warning(f"위험도 분석 시도 {attempt + 1} 실패: {e}")
                if attempt < 2:
                    continue
            
        logger.error("모든 위험도 분석 시도 실패")
        # 최소한의 기본 구조만 반환 (완전 실패 방지)
        return self._get_fallback_result().detail_analysis
    
    def _is_valid_parsed_result(self, detail_analysis) -> bool:
        """
        파싱 결과 유효성 검증 - 모든 카테고리의 제목과 내용이 있어야 통과
        
        Args:
            detail_analysis: DetailAnalysisResult 객체
            
        Returns:
            bool: 파싱 성공 여부
        """
        if not detail_analysis:
            logger.warning("detail_analysis가 None임")
            return False
        
        try:
            # 4개 카테고리 검증
            categories = [
                ('basic_info', detail_analysis.basic_info),
                ('rights_info', detail_analysis.rights_info), 
                ('building_info', detail_analysis.building_info),
                ('legal_info', detail_analysis.legal_info)
            ]
            
            for category_name, category in categories:
                # 카테고리 객체가 없으면 실패
                if not category:
                    logger.warning(f"{category_name} 카테고리가 없음")
                    return False
                
                # 제목이 없거나 기본값만 있으면 실패
                if not category.title or category.title.strip() in ['', '기본 정보 확인', '권리관계 확인', '건축물 확인', '법령 준수 확인']:
                    logger.warning(f"{category_name} 제목이 없거나 기본값임: '{category.title}'")
                    return False
                
                # 내용이 없거나 너무 짧으면 실패
                if not category.content or len(category.content.strip()) < 15:
                    logger.warning(f"{category_name} 내용이 없거나 너무 짧음 (길이: {len(category.content.strip()) if category.content else 0})")
                    return False
            
            logger.info("모든 카테고리 파싱 검증 통과")
            return True
            
        except AttributeError as e:
            logger.warning(f"파싱 검증 중 속성 오류: {e}")
            return False
        except Exception as e:
            logger.error(f"파싱 검증 중 예외 발생: {e}")
            return False
    
    def _parse_detail_analysis_response_with_categories(self, response_text: str, category_risks: Dict[str, RiskLevel]):
        """LLM 상세 분석 응답 파싱 (카테고리별 위험도 포함) - 디버깅 추가"""
        try:
            # 🔍 디버깅: LLM 응답 전체를 로그로 출력
            # logger.info("=== LLM 응답 전체 ===")
            # logger.info(response_text[:1000] + "..." if len(response_text) > 1000 else response_text)
            # logger.info("=== LLM 응답 끝 ===")
            
            # 기본값 설정
            result = {
                'basic_info_title': '기본 정보 확인',
                'basic_info_content': '',
                'rights_info_title': '권리관계 확인',
                'rights_info_content': '',
                'building_info_title': '건축물 확인',
                'building_info_content': '',
                'legal_info_title': '법령 준수 확인',
                'legal_info_content': ''
            }
            
            # 섹션별 내용 추출
            sections = [
            ('basic_info', r'(?:^|\n)\s*(?:#{1,6}\s*)?1[.)]?\s*기본정보\s*분석(.*?)(?=(?:^|\n)\s*(?:#{1,6}\s*)?2[.)]|\Z)'),
            ('rights_info', r'(?:^|\n)\s*(?:#{1,6}\s*)?2[.)]?\s*권리관계\s*분석(.*?)(?=(?:^|\n)\s*(?:#{1,6}\s*)?3[.)]|\Z)'),
            ('building_info', r'(?:^|\n)\s*(?:#{1,6}\s*)?3[.)]?\s*건축관련\s*분석(.*?)(?=(?:^|\n)\s*(?:#{1,6}\s*)?4[.)]|\Z)'),
            ('legal_info', r'(?:^|\n)\s*(?:#{1,6}\s*)?4[.)]?\s*법령위험\s*분석(.*?)(?=(?:^|\n)\s*(?:#{1,6}\s*)?\d[.)]|\Z)')
        ]
            
            for section_name, pattern in sections:
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    section_content = match.group(1).strip()
                    
                    # 🔍 디버깅: 각 섹션별 매칭 내용 출력
                    logger.info(f"=== {section_name} 섹션 매칭 ===")
                    logger.info(section_content[:300] + "..." if len(section_content) > 300 else section_content)
                    
                    # 제목 추출
                    title_match = re.search(r'(?m)^제목:\s*(.+)$', section_content)
                    if title_match:
                        title = title_match.group(1).strip()
                        
                        result[f'{section_name}_title'] = title
                        logger.info(f"제목 추출 성공: {title}")
                    else:
                        logger.warning(f"{section_name} 제목 추출 실패")
                    
                    # 내용 추출 (** 제거 개선)
                    content_match = re.search(r'내용:\s*(.+?)(?:\n\s*(?:제목:|$)|\Z)', section_content, re.DOTALL | re.MULTILINE)
                    if content_match:
                        content = content_match.group(1).strip()
                        # 내용에서 ** 제거 (마크다운 볼드 제거)
                        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # **텍스트** -> 텍스트
                        content = content.strip()  # 공백 제거
                        
                        result[f'{section_name}_content'] = content
                        logger.info(f"내용 추출 성공: {content[:100]}...")
                    else:
                        logger.warning(f"{section_name} 내용 추출 실패")
                else:
                    logger.warning(f"{section_name} 섹션 전체 매칭 실패")

            
            basic_info = CategoryAnalysisResult(
                title=result['basic_info_title'],
                content=result['basic_info_content'],
                risk_level=category_risks['basic_info']
            )
            
            rights_info = CategoryAnalysisResult(
                title=result['rights_info_title'],
                content=result['rights_info_content'],
                risk_level=category_risks['rights_info']
            )
            
            building_info = CategoryAnalysisResult(
                title=result['building_info_title'],
                content=result['building_info_content'],
                risk_level=category_risks['building_info']
            )
            
            legal_info = CategoryAnalysisResult(
                title=result['legal_info_title'],
                content=result['legal_info_content'],
                risk_level=category_risks['legal_info']
            )
            
            return DetailAnalysisResult(
                basic_info=basic_info,
                rights_info=rights_info,
                building_info=building_info,
                legal_info=legal_info
            )
            
        except Exception as e:
            logger.error(f"카테고리별 상세 분석 응답 파싱 실패: {e}")
            import traceback
            traceback.print_exc()
            return None  # improve_model 방식으로 None 반환
    
    
    def _generate_risk_message(self, risk_level: RiskLevel) -> str:
        """위험도별 메시지 생성"""
        messages = {
            RiskLevel.SAFE: "이 매물은 안전한 상황입니다",
            RiskLevel.WARN: "이 매물은 주의가 필요합니다", 
            RiskLevel.DANGER: "이 매물은 위험 상황입니다"
        }
        return messages.get(risk_level, "분석을 완료했습니다")
    
    def _format_user_info(self, user_info) -> str:
        """사용자 정보 포맷팅"""
        return f"사용자 ID: {user_info.user_id}, 유형: {'임대인' if user_info.user_type == 'landlord' else '임차인'}"
    
    def _format_property_info(self, property_info) -> str:
        """매물 정보 포맷팅"""
        return f"""
매물 ID: {property_info.home_id}
주소: {property_info.address}
등록자: {property_info.registered_user_name}
보증금: {f'{property_info.deposit_price:,}원' if property_info.deposit_price else '0원'}
"""
    
    def _format_registry_data(self, data) -> str:
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
    
    def _format_building_data(self, data) -> str:
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
    
    def _get_fallback_result(self):
        """오류시 기본 결과"""
        
        # 기본 카테고리별 위험도
        default_category_risks = {
            'basic_info': RiskLevel.WARN,
            'rights_info': RiskLevel.WARN,
            'building_info': RiskLevel.WARN,
            'legal_info': RiskLevel.WARN
        }
        
        # fallback 메서드 호출하지 않고 직접 생성
        basic_info = CategoryAnalysisResult(
            title="기본 정보 확인",
            content="분석 중 오류가 발생했습니다.",
            risk_level=default_category_risks['basic_info']
        )
        
        rights_info = CategoryAnalysisResult(
            title="권리관계 확인",
            content="분석 중 오류가 발생했습니다.",
            risk_level=default_category_risks['rights_info']
        )
        
        building_info = CategoryAnalysisResult(
            title="건축물 확인",
            content="분석 중 오류가 발생했습니다.",
            risk_level=default_category_risks['building_info']
        )
        
        legal_info = CategoryAnalysisResult(
            title="법령 준수 확인",
            content="분석 중 오류가 발생했습니다.",
            risk_level=default_category_risks['legal_info']
        )
        
        detail_analysis = DetailAnalysisResult(
            basic_info=basic_info,
            rights_info=rights_info,
            building_info=building_info,
            legal_info=legal_info
        )
        
        return RiskAnalysisResult(
            risk_level=RiskLevel.WARN,
            risk_message="분석 중 오류가 발생했습니다",
            detail_analysis=detail_analysis
        )