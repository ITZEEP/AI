"""
AI 기반 특약 생성 모델 - Gemini 2.5 Pro 전용
"""
import os
import sys
import re
import traceback
from typing import List, Dict, Optional
from dotenv import load_dotenv

# 프로젝트 경로 설정
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# 로거 설정
load_dotenv()
from config.logger_config import get_logger
logger = get_logger(__name__)
from config.gemini_retry import retry_gemini_api

# clause_report에서 파서 import
try:
    from generators.clause_report import ClauseDataParser, ClauseData, OwnerPrecheck, TenantPrecheck
    logger.info("clause_report 모듈에서 클래스 import 성공")
except ImportError as e:
    logger.error(f"clause_report import 실패: {e}")


# law_system import (있으면 사용, 없으면 무시)
try:
    from law_system.law_vectorstore import get_law_vectorstore, search_law
    logger.info("law_system module imported successfully")
    LAW_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"law_system import failed: {e}")
    LAW_SYSTEM_AVAILABLE = False


class ClauseGenerationModel:
    """AI 기반 특약 생성 모델 - Gemini 2.5 Pro 전용"""
    
    def __init__(self, model_name: str = "gemini-2.5-pro", temperature: float = 0.07):
        """
        Args:
            model_name: 모델명 (기본값: gemini-2.5-pro)
            temperature: 생성 온도 (0.0-1.0)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._setup_llm()
        self.vectorstore = self._setup_vectorstore()
        self.parser = ClauseDataParser()
        
    def _setup_llm(self):
        """LLM 모델 설정"""
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature
            )
            logger.info(f"Gemini LLM initialized: {self.model_name}")
            return llm
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            raise
    
    def _setup_vectorstore(self):
        """벡터스토어 설정"""
        if not LAW_SYSTEM_AVAILABLE:
            logger.warning("Working without RAG - law_system not available")
            return None
        
        try:
            vectorstore = get_law_vectorstore()
            if vectorstore:
                logger.info("Vectorstore connected successfully")
                return vectorstore
            else:
                logger.warning("Vectorstore is None")
                return None
        except Exception as e:
            logger.error(f"Vectorstore connection failed: {e}")
            return None
    
    @retry_gemini_api(max_retries=5, initial_delay=2.0, backoff_multiplier=1.5)
    def _call_gemini_api(self, chain, invoke_params):
        """
        Gemini API 호출 래퍼 메서드 (재시도 로직 적용)
        
        Args:
            chain: LangChain 체인
            invoke_params: invoke에 전달할 파라미터
        
        Returns:
            API 호출 결과
        """
        logger.debug("Gemini API 호출 시작")
        result = chain.invoke(invoke_params)
        logger.debug("Gemini API 호출 성공")
        return result
    
    def generate_initial_clauses(self, 
                               owner_data: Dict,
                               tenant_data: Dict,
                               ocr_data: Optional[Dict] = None) -> List[ClauseData]:
        """
        초기 특약 6개 생성
        
        Args:
            owner_data: 임대인 사전조사 JSON
            tenant_data: 임차인 사전조사 JSON
            ocr_data: OCR 결과 JSON (선택사항)
            
        Returns:
            List[ClauseData]: 생성된 특약 리스트 (6개)
        """
        try:
            logger.info("초기 특약 생성 시작")
            
            # 데이터 파싱
            owner = self.parser.parse_owner_precheck(owner_data)
            tenant = self.parser.parse_tenant_precheck(tenant_data)
            ocr = self.parser.parse_ocr_result(ocr_data) if ocr_data else None
            
            # 컨텍스트 생성
            owner_context = self.parser.create_context_from_owner(owner)
            tenant_context = self.parser.create_context_from_tenant(tenant)
            ocr_context = self.parser.create_context_from_ocr(ocr) if ocr else ""
            
            # 법령 검색 (RAG)
            law_context = self._search_relevant_laws(owner, tenant)
            
            # 특약 생성
            clauses = self._create_initial_clauses_with_retry(
                owner_context, tenant_context, ocr_context, law_context
            )
            
            # 정확히 6개만 반환
            if len(clauses) > 6:
                clauses = clauses[:6]
            
            logger.info(f"초기 특약 {len(clauses)}개 생성 완료")
            return clauses
            
        except Exception as e:
            logger.error(f"특약 생성 중 오류: {e}")
            traceback.print_exc()
            return []
    
    def _search_relevant_laws(self, owner: OwnerPrecheck, tenant: TenantPrecheck) -> str:
        """관련 법령 검색"""
        if not LAW_SYSTEM_AVAILABLE or not self.vectorstore:
            return "\n# 법령 검색을 사용할 수 없습니다.\n"
        
        try:
            # 검색 쿼리 생성
            queries = [
                f"{owner.rentType} 임대차 계약 특약",
                "주택임대차보호법 임차인 권리",
                "전세 보증금 보호",
            ]
            
            if tenant.hasPet:
                queries.append("반려동물 임대차 특약")
            if owner.requireRentGuaranteeInsurance:
                queries.append("임대차 보증보험")
            if tenant.facilityRepairNeeded:
                queries.append("임대 시설 수리 책임")
            
            # 법령 검색 수행
            all_results = []
            for query in queries:
                results = search_law(query, k=3)
                all_results.extend(results)
            
            # 중복 제거 및 포맷팅
            seen = set()
            law_context = "\n# 관련 법령:\n"
            for result in all_results:
                law_name = result.get('law_name', '')
                article = result.get('article', '')
                key = f"{law_name}-{article}"
                
                if key not in seen:
                    seen.add(key)
                    content = result.get('content', '')[:200]
                    law_context += f"\n## {law_name} {article}\n{content}...\n"
            
            return law_context
            
        except Exception as e:
            logger.warning(f"법령 검색 실패: {e}")
            return "\n# 법령 검색 중 오류가 발생했습니다.\n"
    
    def _create_initial_clauses_with_retry(self, 
                                         owner_context: str,
                                         tenant_context: str,
                                         ocr_context: str,
                                         law_context: str) -> List[ClauseData]:
        """LLM을 사용한 초기 특약 생성"""
        
        # 특약 생성 프롬프트
        prompt = PromptTemplate.from_template("""
당신은 부동산 임대차 계약 전문가입니다.
다음 정보를 바탕으로 **정확히 6개의 특약**을 추천해주세요.

{owner_context}

{tenant_context}

{ocr_context}

{law_context}

# 특약 생성 규칙:
1. 반드시 6개의 특약만 생성하세요
2. 임대인과 임차인의 조건을 모두 고려하세요
3. 기존 OCR 특약과 중복되지 않는 새로운 특약을 만드세요
4. 구체적이고 실행 가능한 내용으로 작성하세요
5. 법령에 근거한 특약을 우선시하세요
6. 양측의 이익을 균형있게 반영하세요
7. 전세/월세 유형에 맞는 특약을 생성하세요
8. 각 특약 내용은 1-2문장으로 간결하게 작성하세요
9. 핵심 내용만 포함하고 불필요한 설명은 제외하세요

# 특별 고려사항:
- 반려동물, 흡연, 중도퇴거 등 구체적 조건 반영
- 보증보험, 대출 계획 등 금융 관련 사항 고려
- 설비 수리, 원상복구 등 관리 책임 명확화
- 재계약, 기간 연장 등 장기 거주 조건 반영

# 중요: 아래 형식을 정확히 따라주세요. 다른 내용은 추가하지 마세요.

## 1번 특약
제목: [특약 제목]
내용: [1-2문장으로 구체적이고 간결한 특약 조항 내용]

## 2번 특약
제목: [특약 제목]
내용: [1-2문장으로 구체적이고 간결한 특약 조항 내용]

## 3번 특약
제목: [특약 제목]
내용: [1-2문장으로 구체적이고 간결한 특약 조항 내용]

## 4번 특약
제목: [특약 제목]
내용: [1-2문장으로 구체적이고 간결한 특약 조항 내용]

## 5번 특약
제목: [특약 제목]
내용: [1-2문장으로 구체적이고 간결한 특약 조항 내용]

## 6번 특약
제목: [특약 제목]
내용: [1-2문장으로 구체적이고 간결한 특약 조항 내용]
""")
        
        # LLM 체인 구성
        chain = prompt | self.llm | StrOutputParser()
        
        # 특약 생성 (재시도 로직 적용)
        try:
            logger.info("Gemini API 호출 중... (재시도 로직 활성화)")
            
            # 재시도 로직이 적용된 API 호출
            result = self._call_gemini_api(chain, {
                "owner_context": owner_context,
                "tenant_context": tenant_context,
                "ocr_context": ocr_context,
                "law_context": law_context
            })
            
            logger.info("Gemini API 호출 완료")
            
            # 결과 파싱 - clause_report의 파서 사용
            clauses = self.parser.parse_llm_clauses_output(result)
            
            if not clauses:
                logger.warning("파싱된 특약이 없습니다. LLM 응답 확인 필요")
                logger.debug(f"LLM 응답: {result[:300]}...")
            
            return clauses
            
        except Exception as e:
            logger.error(f"LLM 특약 생성 최종 실패: {e}")
            logger.error("모든 재시도 시도가 실패했습니다.")
            return []
    
    