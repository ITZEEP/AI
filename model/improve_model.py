"""
AI 기반 특약 개선 모델 - Gemini 2.5 Pro 전용
대화 내용을 분석하여 특약을 개선하고 임대인/임차인 입장에서 평가
"""
import os
import sys
import re
import traceback
from typing import Optional
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

# clause_improvement에서 데이터 클래스 import
try:
    from generators.improve_report import (
        ClauseImprovementRequest, ImprovedClause, ClauseImprovementParser, AssessmentLevel
    )
    logger.info("clause_improvement 모듈에서 클래스 import 성공")
except ImportError as e:
    logger.error(f"clause_improvement import 실패: {e}")

# law_system import (있으면 사용, 없으면 무시)
try:
    from law_system.law_vectorstore import get_law_vectorstore, search_law
    logger.info("law_system module imported successfully")
    LAW_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"law_system import failed: {e}")
    LAW_SYSTEM_AVAILABLE = False


class ClauseImprovementModel:
    """AI 기반 특약 개선 모델 - Gemini 2.5 Pro 전용"""
    
    def __init__(self, model_name: str = "gemini-2.5-pro", temperature: float = 0.1):
        """
        Args:
            model_name: 모델명 (기본값: gemini-2.5-pro)
            temperature: 생성 온도 (개선의 일관성을 위해 낮은 값)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._setup_llm()
        self.vectorstore = self._setup_vectorstore()
        self.parser = ClauseImprovementParser()
        
    def _setup_llm(self):
        """LLM 모델 설정"""
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
            )
            logger.info(f"Improvement LLM initialized: {self.model_name}")
            return llm
        except Exception as e:
            logger.error(f"Improvement LLM initialization failed: {e}")
            raise
    
    def _setup_vectorstore(self):
        """벡터스토어 설정"""
        if not LAW_SYSTEM_AVAILABLE:
            logger.warning("Working without RAG - law_system not available")
            return None
        
        try:
            vectorstore = get_law_vectorstore()
            if vectorstore:
                logger.info("Vectorstore connected successfully for improvement")
                return vectorstore
            else:
                logger.warning("Vectorstore is None")
                return None
        except Exception as e:
            logger.error(f"Vectorstore connection failed: {e}")
            return None
    
    def _search_relevant_laws(self, request: ClauseImprovementRequest) -> str:
        """관련 법령 검색 (동적 키워드 기반)"""
        if not LAW_SYSTEM_AVAILABLE or not self.vectorstore:
            return "\n# 법령 검색을 사용할 수 없습니다.\n"
        
        try:
            queries = []
            
            # 1. 기본 임대차 법령
            if request.owner_data:
                rent_type = request.owner_data.get('rentType', '')
                queries.append(f"{rent_type} 임대차 특약 법령")
            
            # 2. 특약 제목과 내용에서 핵심 키워드 추출
            recent_title = request.recent_data.title
            recent_content = request.recent_data.content
            recent_messages = request.recent_data.messages
            
            # 전체 텍스트 결합
            full_text = f"{recent_title} {recent_content} {recent_messages}"
            
            # 3. 법령 관련 키워드 사전 정의
            law_keywords = {
                # 동물 관련
                "반려동물|애완동물|펜트|강아지|고양이|동물": ["반려동물 임대차", "애완동물 사육 규정"],
                
                # 수리/보수 관련  
                "수리|보수|교체|고장|파손|설비|시설": ["임대차 수리 책임", "시설 보수 의무", "설비 교체 법령"],
                
                # 금전 관련
                "보증금|임대료|월세|전세|연체|이자": ["임대차 보증금", "임대료 연체 법령", "전세금 보호"],
                
                # 청소/원상복구 관련
                "청소|원상복구|복구|도배|장판|벽지": ["원상복구 의무", "임대차 복구 범위"],
                
                # 흡연 관련
                "흡연|담배|금연": ["임대차 흡연 금지", "주거 환경 보호"],
                
                # 소음 관련  
                "소음|층간소음|악기|음악|파티": ["층간소음 규제", "주거 평온권"],
                
                # 전대/용도변경 관련
                "전대|임대|용도변경|상업|사업": ["전대차 금지", "용도변경 제한"],
                
                # 계약 관련
                "계약|연장|갱신|해지|중도|퇴거": ["임대차 계약 해지", "계약 갱신 법령"],
                
                # 보험 관련
                "보험|보증보험": ["임대차 보증보험"],
                
                # 입주/퇴거 관련
                "입주|퇴거|이사|확정일자": ["확정일자 법령", "임대차 신고"],
                
                # 공용부분 관련
                "공용|베란다|옥상|주차": ["공용부분 사용", "부대시설 이용"],
                
                # 기타
                "화재|안전|방범": ["주거 안전 의무"],
                "프라이버시|개인정보": ["임차인 프라이버시권"]
            }
            
            # 4. 키워드 매칭으로 관련 법령 쿼리 생성
            import re
            for pattern, law_queries in law_keywords.items():
                if re.search(pattern, full_text, re.IGNORECASE):
                    queries.extend(law_queries)
            
            # 5. 사전조사 정보 기반 추가 검색
            if request.owner_data:
                if request.owner_data.get('requireRentGuaranteeInsurance'):
                    queries.append("임대차 보증보험 의무")
                if request.owner_data.get('hasPenalty'):
                    queries.append("임대차 위약금 법령")
            
            if request.tenant_data:
                if request.tenant_data.get('hasPet'):
                    queries.append("반려동물 임대차 법령")
                if request.tenant_data.get('earlyTerminationRisk'):
                    queries.append("임대차 중도해지")
            
            # 6. 기본 필수 법령 (항상 포함)
            essential_queries = [
                "주택임대차보호법 특약",
                "임대차 계약 조건",
                "임차인 권리 의무"
            ]
            queries.extend(essential_queries)
            
            # 7. 중복 제거 및 제한
            unique_queries = list(dict.fromkeys(queries))[:8]  # 최대 8개
            
            logger.debug(f"법령 검색 쿼리: {unique_queries}")
            
            # 8. 법령 검색 수행
            all_results = []
            for query in unique_queries:
                try:
                    results = search_law(query, k=2)  # 각 쿼리당 2개씩
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"법령 검색 실패 - 쿼리: {query}, 오류: {e}")
                    continue
            
            # 9. 결과 포맷팅
            seen = set()
            law_context = "\n# 관련 법령 참고사항:\n"
            
            for result in all_results[:8]:  # 최대 8개 법령
                law_name = result.get('law_name', '')
                article = result.get('article', '')
                key = f"{law_name}-{article}"
                
                if key not in seen and law_name and article:
                    seen.add(key)
                    content = result.get('content', '')[:120]  # 120자로 제한
                    law_context += f"\n## {law_name} {article}\n{content}...\n"
            
            if len(seen) == 0:
                law_context += "- 관련 법령을 찾지 못했습니다.\n"
            
            return law_context
            
        except Exception as e:
            logger.warning(f"법령 검색 중 전체 오류: {e}")
            return "\n# 법령 검색 중 오류가 발생했습니다.\n"
    
    @retry_gemini_api(max_retries=5, initial_delay=2.0, backoff_multiplier=1.5)
    def _call_gemini_api_for_improvement(self, chain, invoke_params):
        """
        Gemini API 호출 래퍼 메서드 (개선용, 재시도 로직 적용)
        
        Args:
            chain: LangChain 체인
            invoke_params: invoke에 전달할 파라미터
        
        Returns:
            API 호출 결과
        """
        logger.debug("Gemini API 호출 시작 (특약 개선)")
        result = chain.invoke(invoke_params)
        logger.debug("Gemini API 호출 성공 (특약 개선)")
        return result
    
    def improve_clause(self, request: ClauseImprovementRequest) -> Optional[ImprovedClause]:
        """
        대화 내용을 바탕으로 특약 개선
        
        Args:
            request: 특약 개선 요청 데이터
            
        Returns:
            ImprovedClause: 개선된 특약 및 평가 결과
        """
        try:
            logger.info(f"특약 개선 시작 - 라운드: {request.round}, 특약: {request.order}번")
            
            # 컨텍스트 생성
            context = self.parser.create_context_for_llm(request)
            
            # 법령 검색 (RAG)
            law_context = self._search_relevant_laws(request)
            
            # 특약 개선 및 평가 수행
            improved_clause = self._improve_clause_with_retry(request, context, law_context)
            
            if improved_clause:
                logger.info(f"특약 개선 완료 - 라운드: {request.round}, 특약: {request.order}번")
            else:
                logger.error("특약 개선 실패")
            
            return improved_clause
            
        except Exception as e:
            logger.error(f"특약 개선 중 오류: {e}")
            traceback.print_exc()
            return None
    
    def _improve_clause_with_retry(self, 
                                 request: ClauseImprovementRequest, 
                                 context: str,
                                 law_context: str) -> Optional[ImprovedClause]:
        """LLM을 사용한 특약 개선 및 평가"""
        
        prompt = PromptTemplate.from_template("""
당신은 부동산 임대차 계약 전문가입니다.
임대인과 임차인의 대화 내용을 분석하여 특약을 개선하고, 각 당사자 입장에서 평가해주세요.

{context}

{law_context}

# 개선 목표:
1. 대화에서 나타난 양측의 요구사항과 우려사항 반영
2. 구체적이고 실행 가능한 조건으로 수정
3. 법적으로 문제가 없고 공정한 내용으로 개선
4. 모호한 표현을 명확하고 구체적으로 수정
5. 양측이 모두 수용할 수 있는 합리적인 절충안 도출

# 평가 기준:
- **안심**: 해당 당사자에게 유리하거나 공정하고 균형적인 특약
- **주의**: 해당 당사자에게 불리하거나 위험할 수 있는 특약

# 평가 시 고려사항:
1. 대화에서 표현된 각 당사자의 입장과 우려사항
2. 개선된 특약이 각 당사자에게 미치는 영향
3. 법적 권리와 의무의 균형
4. 경제적 부담과 실행 가능성
5. 분쟁 발생 시 각 당사자의 보호 정도

## 우리 시스템의 계약 구조 (중요!):
- **전세 계약**: 보증금만 존재 (월세 없음)
- **월세 계약**: 보증금 + 월세 존재
- **계약금은 우리 시스템에 아예 없음** (시퀀스에 포함되지 않음)
- **위약금은 오직 보증금 기준으로만 산정함**

# 답변 작성 가이드:
- 대화 내용을 충분히 반영한 현실적인 개선안 제시
- 구체적인 수치, 조건, 절차 등을 명확히 기술
- 실무적이고 자연스러운 어조로 작성
- "~할 수 있습니다", "~하기로 합니다" 등 계약서 문체 사용

# 출력 형식 (정확히 이 형식을 지켜주세요. 볼드나 다른 마크다운 사용 금지):

## 개선된 특약 제목
[대화 내용을 반영한 구체적이고 명확한 제목]

## 개선된 특약 내용
[대화에서 합의된 내용을 반영한 구체적이고 실행 가능한 특약 조항. 1-2문장으로 작성하되 모든 중요한 조건을 포함]

## 임대인 평가
등급: 안심
이유: 개선된 특약이 임대인에게 미치는 영향을 2-3문장으로 설명

## 임차인 평가
등급: 주의
이유: 개선된 특약이 임차인에게 미치는 영향을 2-3문장으로 설명

중요: **볼드**, *이탤릭* 등 마크다운 문법을 사용하지 마세요. 순수 텍스트로만 작성해주세요.
""")
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            logger.debug(f"특약 개선 API 호출 - 라운드: {request.round}")
            
            # 재시도 로직이 적용된 API 호출
            result = self._call_gemini_api_for_improvement(chain, {
                "context": context,
                "law_context": law_context
            })
            
            logger.debug(f"특약 개선 API 완료 - 라운드: {request.round}")
            
            # 결과 파싱
            improved_clause = self._parse_improvement_result(request, result)
            return improved_clause
            
        except Exception as e:
            logger.error(f"특약 개선 최종 실패: {e}")
            return None
    
    def _parse_improvement_result(self, 
                                request: ClauseImprovementRequest, 
                                llm_output: str) -> Optional[ImprovedClause]:
        """LLM 개선 결과 파싱"""
        
        try:
            logger.debug("개선 결과 파싱 시작")
            
            # 초기값 설정
            title = ""
            content = ""
            owner_assessment = AssessmentLevel.SAFE
            owner_reason = ""
            tenant_assessment = AssessmentLevel.SAFE
            tenant_reason = ""
            
            lines = llm_output.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                # 섹션 구분
                if "## 개선된 특약 제목" in line:
                    current_section = "title"
                elif "## 개선된 특약 내용" in line:
                    current_section = "content"
                elif "## 임대인 평가" in line:
                    current_section = "owner"
                elif "## 임차인 평가" in line:
                    current_section = "tenant"
                # 내용 파싱
                elif line and not line.startswith("##"):
                    if current_section == "title" and not title:
                        title = line
                    elif current_section == "content" and not content:
                        content = line
                    elif current_section == "owner":
                        if line.startswith("등급:"):
                            grade = line.replace("등급:", "").strip()
                            if "주의" in grade:
                                owner_assessment = AssessmentLevel.CAUTION
                        elif line.startswith("이유:"):
                            owner_reason = line.replace("이유:", "").strip()
                    elif current_section == "tenant":
                        if line.startswith("등급:"):
                            grade = line.replace("등급:", "").strip()
                            if "주의" in grade:
                                tenant_assessment = AssessmentLevel.CAUTION
                        elif line.startswith("이유:"):
                            tenant_reason = line.replace("이유:", "").strip()
            
            # 필수 필드 검증
            if not title or not content:
                logger.error(f"필수 필드 누락 - 제목: {bool(title)}, 내용: {bool(content)}")
                return None
            
            return ImprovedClause(
                round=request.round,
                order=request.order,
                title=title,
                content=content,
                owner_assessment=owner_assessment,
                owner_reason=owner_reason or "평가 정보가 부족합니다.",
                tenant_assessment=tenant_assessment,
                tenant_reason=tenant_reason or "평가 정보가 부족합니다."
            )
            
        except Exception as e:
            logger.error(f"개선 결과 파싱 실패: {e}")
            traceback.print_exc()
            return None

