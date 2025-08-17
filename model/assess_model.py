"""
특약 평가 모델 - 임대인/임차인 입장별 특약 안전성 평가
생성된 특약을 각 당사자 입장에서 '안심' 또는 '주의' 등급으로 평가
Gemini 2.5 Pro 기반 병렬 처리 최적화
"""
import os
import sys
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
import time
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

# 표준조항 import
from config.standard_clauses import STANDARD_CLAUSES

# clause_report에서 파서 및 데이터 클래스 import
try:
    from generators.clause_report import (
        ClauseDataParser, ClauseData, ClauseAssessment, AssessmentLevel
    )
    logger.info(" clause_report 모듈에서 클래스 import 성공")
except ImportError as e:
    logger.error(f" clause_report import 실패: {e}")


class ClauseAssessmentModel:
    """특약 평가 모델 - 임대인/임차인 입장별 안전성 평가 (Gemini 2.5 Pro + 병렬 처리)"""
    
    def __init__(self, model_name: str = "gemini-2.5-pro", temperature: float = 0.05):
        """
        Args:
            model_name: 모델명 (기본값: gemini-2.5-pro)
            temperature: 생성 온도 (평가 일관성을 위해 낮은 값)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._setup_llm()
        # 공통 파서 사용 (중복 제거)
        self.parser = ClauseDataParser()
        
    def _setup_llm(self):
        """LLM 모델 설정"""
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
            )
            logger.info(f" Assessment LLM initialized: {self.model_name}")
            return llm
        except Exception as e:
            logger.error(f" Assessment LLM initialization failed: {e}")
            raise
    
    @retry_gemini_api(max_retries=5, initial_delay=2.0, backoff_multiplier=1.5)
    def _call_gemini_api_for_assessment(self, chain, invoke_params):
        """
        Gemini API 호출 래퍼 메서드 (평가용, 재시도 로직 적용)
        
        Args:
            chain: LangChain 체인
            invoke_params: invoke에 전달할 파라미터
        
        Returns:
            API 호출 결과
        """
        logger.debug("Gemini API 호출 시작 (평가)")
        result = chain.invoke(invoke_params)
        logger.debug("Gemini API 호출 성공 (평가)")
        return result
    
    def assess_clauses(self,
                      clauses: List[ClauseData],
                      owner_data: Dict,
                      tenant_data: Dict,
                      max_workers: int = 3) -> List[ClauseAssessment]:
        """
        특약 리스트를 병렬로 평가 (Gemini 2.5 Pro 최적화)
        
        Args:
            clauses: 평가할 특약 리스트
            owner_data: 임대인 사전조사 JSON
            tenant_data: 임차인 사전조사 JSON
            max_workers: 동시 처리할 최대 작업 수 (기본 3개)
        """
        try:
            logger.info(f"Gemini 2.5 Pro로 {len(clauses)}개 특약 병렬 평가 시작 (workers: {max_workers}, 재시도 로직 활성화)")
            start_time = time.time()
            
            # 데이터 파싱
            owner = self.parser.parse_owner_precheck(owner_data)
            tenant = self.parser.parse_tenant_precheck(tenant_data)
            
            # 컨텍스트 생성
            owner_context = self.parser.create_context_from_owner(owner)
            tenant_context = self.parser.create_context_from_tenant(tenant)
            
            # 특약을 그룹으로 나누기 (Gemini 2.5 Pro는 더 큰 배치 처리 가능)
            clause_groups = self._split_clauses_into_groups(clauses, max_workers)
            
            # ThreadPoolExecutor로 병렬 처리
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 각 그룹을 병렬로 처리
                futures = []
                for group in clause_groups:
                    future = executor.submit(
                        self._assess_clause_group_with_retry,
                        group, owner_context, tenant_context
                    )
                    futures.append(future)
                
                # 결과 수집
                all_assessments = []
                for future in futures:
                    group_assessments = future.result()
                    all_assessments.extend(group_assessments)
            
            # ID 순으로 정렬
            all_assessments.sort(key=lambda x: x.clause_id)
            
            end_time = time.time()
            logger.info(f"병렬 평가 완료: {len(all_assessments)}개 ({end_time - start_time:.1f}초)")
            return all_assessments
            
        except Exception as e:
            logger.error(f"병렬 평가 중 오류: {e}")
            traceback.print_exc()
            return []
    
    def _split_clauses_into_groups(self, clauses: List[ClauseData], max_workers: int) -> List[List[ClauseData]]:
        """특약을 그룹으로 나누기"""
        
        if len(clauses) <= max_workers:
            # 특약이 적으면 각각 개별 처리
            return [[clause] for clause in clauses]
        
        # 그룹 크기 계산
        group_size = (len(clauses) + max_workers - 1) // max_workers
        
        groups = []
        for i in range(0, len(clauses), group_size):
            group = clauses[i:i + group_size]
            groups.append(group)
        
        logger.info(f"특약 그룹 분할: {len(groups)}개 그룹, 그룹당 평균 {group_size}개")
        return groups
    
    def _assess_clause_group_with_retry(self,
                                      clause_group: List[ClauseData],
                                      owner_context: str,
                                      tenant_context: str) -> List[ClauseAssessment]:
        """특약 그룹 평가 (개별 스레드에서 실행)"""
        
        try:
            # 각 그룹에 대해 새로운 LLM 인스턴스 생성 (스레드 안전성)
            group_llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
            )
            
            if len(clause_group) == 1:
                # 단일 특약 처리
                return self._assess_single_clause_with_retry(
                    clause_group[0], owner_context, tenant_context, group_llm
                )
            else:               
                for batch_attempt in range(3):
                    try:
                        # 배치 평가 실행
                        assessments = self._assess_multiple_clauses_with_retry(
                            clause_group, owner_context, tenant_context, group_llm
                        )
                        if len(assessments) == len(clause_group):
                            return assessments
                    except Exception as e:
                        logger.error(f"배치 평가 시도 {batch_attempt + 1} 오류: {e}")
                
                fallback_assessments = []
                for clause in clause_group:
                    single_assessment = self._assess_single_clause_with_retry(
                        clause, owner_context, tenant_context, group_llm
                    )
                    if single_assessment:
                        fallback_assessments.extend(single_assessment)
                return fallback_assessments
                    
        except Exception as e:
            logger.error(f"그룹 평가 실패: {e}")
            return []
    
    def _assess_single_clause_with_retry(self,
                                       clause: ClauseData,
                                       owner_context: str,
                                       tenant_context: str,
                                       llm) -> List[ClauseAssessment]:
        """단일 특약 평가"""
        
        prompt = PromptTemplate.from_template("""
당신은 부동산 임대차 계약 전문가입니다.
다음 특약을 임대인과 임차인 각각의 입장에서 평가해주세요.

# 평가할 특약:
제목: {clause_title}
내용: {clause_content}

# 임대인 정보:
{owner_context}

# 임차인 정보:
{tenant_context}

# 표준계약서 필수조항 (평가 시 참고):
{standard_clauses}

# 평가 기준:
- **안심**: 해당 당사자에게 유리하거나 공정하고 균형적인 특약
- **주의**: 해당 당사자에게 불리하거나 위험할 수 있는 특약

# 평가 시 고려사항:
1. 각 당사자의 사전조사 내용과 특약의 연관성
2. 법적 권리와 의무의 균형
3. 경제적 손익과 위험 요소
4. 실무적 실행 가능성과 부담 정도
5. 분쟁 발생 시 각 당사자의 보호 정도
6. **표준조항과의 관계 및 적법성 (중요!)**

# 표준조항 관련 평가 기준:
- 표준조항과 중복되거나 모순되는 특약은 **주의** 등급
- 표준조항을 구체화/보완하는 특약은 적절히 평가
- 표준조항 범위를 벗어나는 새로운 내용은 내용에 따라 평가
- 법적 효력이 없거나 무효인 특약은 **주의** 등급

## 우리 시스템의 계약 구조 (중요!):
- **전세 계약**: 보증금만 존재 (월세 없음)
- **월세 계약**: 보증금 + 월세 존재
- **계약금은 우리 시스템에 아예 없음** (시퀀스에 포함되지 않음)
- **위약금은 오직 보증금 기준으로만 산정함**
- **계약금, 잔금 내용은 절대 금지**
- **직거래 계약이므로 중개보수 및 중개 내용 금지**

# 답변 작성 가이드:
- 자연스럽고 실용적인 어조로 작성
- "~할 수 있습니다", "~가 가능합니다" 등 실무적 표현 사용
- 구체적이고 actionable한 내용 포함

# 출력 형식 (정확히 이 형식을 지켜주세요. 볼드나 다른 마크다운 사용 금지):

## 임대인 평가
등급: 안심
이유: 2-3문장으로 간결하게 설명

## 임차인 평가  
등급: 주의
이유: 2-3문장으로 간결하게 설명

중요: **볼드**, *이탤릭* 등 마크다운 문법을 사용하지 마세요. 순수 텍스트로만 작성해주세요.
""")
        
        chain = prompt | llm | StrOutputParser()
        
        try:
            logger.debug(f"단일 특약 평가 시작: {clause.title}")
            
            # 재시도 로직이 적용된 API 호출
            result = self._call_gemini_api_for_assessment(chain, {
                "clause_title": clause.title,
                "clause_content": clause.content,
                "owner_context": owner_context,
                "tenant_context": tenant_context,
                "standard_clauses": STANDARD_CLAUSES
            })
            
            logger.debug(f"단일 특약 평가 완료: {clause.title}")
            
            assessment = self._parse_assessment_result(clause, result)
            return [assessment] if assessment else []
            
        except Exception as e:
            logger.error(f"단일 특약 평가 최종 실패: {e}")
            return []
    
    def _assess_multiple_clauses_with_retry(self,
                                          clause_group: List[ClauseData],
                                          owner_context: str,
                                          tenant_context: str,
                                          llm) -> List[ClauseAssessment]:
        """여러 특약 배치 평가"""
        
        # 그룹 내 특약들을 배치로 처리 (실제 특약 ID 사용)
        clauses_text = ""
        for clause in clause_group:
            clauses_text += f"""
특약 {clause.id}번:
제목: {clause.title}
내용: {clause.content}
"""
        
        # 특약 ID 리스트 생성 (파싱용)
        clause_ids = [clause.id for clause in clause_group]
        clause_ids_str = ", ".join(map(str, clause_ids))
        
        prompt = PromptTemplate.from_template("""
당신은 부동산 임대차 계약 전문가입니다.
다음 {total_count}개 특약을 임대인과 임차인 각각의 입장에서 평가해주세요.

{clauses_text}

# 임대인 정보:
{owner_context}

# 임차인 정보:
{tenant_context}

# 표준계약서 필수조항 (평가 시 참고):
{standard_clauses}

# 평가 기준:
- **안심**: 해당 당사자에게 유리하거나 공정하고 균형적인 특약
- **주의**: 해당 당사자에게 불리하거나 위험할 수 있는 특약

# 평가 시 고려사항:
1. 각 당사자의 사전조사 내용과 특약의 연관성
2. 법적 권리와 의무의 균형
3. 경제적 손익과 위험 요소
4. 실무적 실행 가능성과 부담 정도
5. 분쟁 발생 시 각 당사자의 보호 정도
6. **표준조항과의 관계 및 적법성 (중요!)**

# 표준조항 관련 평가 기준:
- 표준조항과 중복되거나 모순되는 특약은 **주의** 등급
- 표준조항을 구체화/보완하는 특약은 적절히 평가
- 표준조항 범위를 벗어나는 새로운 내용은 내용에 따라 평가
- 법적 효력이 없거나 무효인 특약은 **주의** 등급

## 우리 시스템의 계약 구조 (중요!):
- **전세 계약**: 보증금만 존재 (월세 없음)
- **월세 계약**: 보증금 + 월세 존재
- **계약금은 우리 시스템에 아예 없음** (시퀀스에 포함되지 않음)
- **위약금은 오직 보증금 기준으로만 산정함**
- **계약금, 잔금 내용은 절대 금지**
- **직거래 계약이므로 중개보수 및 중개 내용 금지**

# 답변 작성 가이드:
- 자연스럽고 실용적인 어조로 작성
- "~할 수 있습니다", "~가 가능합니다" 등 실무적 표현 사용
- 구체적이고 actionable한 내용 포함

# 출력 형식 (정확히 이 형식을 지켜주세요. 볼드나 다른 마크다운 사용 금지):
# 평가할 특약들: {clause_ids}

{format_example}

중요: **볼드**, *이탤릭* 등 마크다운 문법을 사용하지 마세요. 순수 텍스트로만 작성해주세요.
""")
        
        # 동적으로 출력 형식 예시 생성
        format_example = ""
        for clause in clause_group:
            format_example += f"""
## 특약 {clause.id}번 평가
임대인: 안심
이유: 2-3문장으로 간결하게 설명
임차인: 주의  
이유: 2-3문장으로 간결하게 설명
"""
        
        chain = prompt | llm | StrOutputParser()
        
        try:
            logger.debug(f"🔄 배치 평가 시작: {len(clause_group)}개 특약 (ID: {clause_ids_str})")
            
            # 재시도 로직이 적용된 API 호출
            result = self._call_gemini_api_for_assessment(chain, {
                "total_count": len(clause_group),
                "clauses_text": clauses_text,
                "owner_context": owner_context,
                "tenant_context": tenant_context,
                "clause_ids": clause_ids_str,
                "format_example": format_example.strip(),
                "standard_clauses": STANDARD_CLAUSES
            })
            
            assessments = self._parse_batch_assessment_result(clause_group, result)
            return assessments
            
        except Exception as e:
            logger.error(f"배치 평가 최종 실패: {e}")
            return []
    
    def _parse_assessment_result(self, clause: ClauseData, llm_output: str) -> Optional[ClauseAssessment]:
        """LLM 평가 결과 파싱 (볼드 마크다운 대응)"""
        
        try:
            # 임대인 평가 파싱
            owner_assessment = AssessmentLevel.SAFE  # 기본값
            owner_reason = ""
            
            # 임차인 평가 파싱
            tenant_assessment = AssessmentLevel.SAFE  # 기본값
            tenant_reason = ""
            
            lines = llm_output.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                # 섹션 구분 (볼드 마크다운 고려)
                if "## 임대인 평가" in line or "**임대인" in line:
                    current_section = "owner"
                elif "## 임차인 평가" in line or "**임차인" in line:
                    current_section = "tenant"
                # 등급 파싱 (여러 형식 대응)
                elif line.startswith("등급:") or line.startswith("**등급:") or ("등급:" in line and current_section):
                    grade = line.replace("등급:", "").replace("*", "").strip()
                    if "주의" in grade:
                        if current_section == "owner":
                            owner_assessment = AssessmentLevel.CAUTION
                        elif current_section == "tenant":
                            tenant_assessment = AssessmentLevel.CAUTION
                # 이유 파싱 (여러 형식 대응)
                elif line.startswith("이유:") or line.startswith("**이유:") or ("이유:" in line and current_section):
                    reason = line.replace("이유:", "").replace("*", "").strip()
                    if current_section == "owner" and reason:
                        owner_reason = reason
                    elif current_section == "tenant" and reason:
                        tenant_reason = reason
                # 볼드 형식의 등급/이유 처리
                elif current_section and ("안심" in line or "주의" in line) and not owner_reason and not tenant_reason:
                    # 등급이 단독으로 나오는 경우
                    grade = line.replace("*", "").strip()
                    if "주의" in grade:
                        if current_section == "owner":
                            owner_assessment = AssessmentLevel.CAUTION
                        elif current_section == "tenant":
                            tenant_assessment = AssessmentLevel.CAUTION
            
            return ClauseAssessment(
                clause_id=clause.id,
                clause_title=clause.title,
                clause_content=clause.content,
                owner_assessment=owner_assessment,
                owner_reason=owner_reason,
                tenant_assessment=tenant_assessment,
                tenant_reason=tenant_reason
            )
            
        except Exception as e:
            logger.error(f"평가 결과 파싱 실패: {e}")
            return None
    
    def _parse_batch_assessment_result(self, clauses: List[ClauseData], llm_output: str) -> List[ClauseAssessment]:
        """배치 평가 결과 파싱 (실제 특약 ID 기반)"""
        
        assessments = []
        
        try:
            logger.debug("LLM 응답 원문 (배치 평가):")
            logger.debug(llm_output[:500] + "..." if len(llm_output) > 500 else llm_output)
            
            # 실제 특약 ID를 사용한 매핑
            clause_id_map = {clause.id: clause for clause in clauses}
            logger.debug(f"처리할 특약 ID들: {list(clause_id_map.keys())}")
            
            # 정규식 패턴 - 실제 특약 ID 기반
            pattern = r'##\s*특약\s*(\d+)번 평가\s*임대인:\s*(안심|주의)\s*이유:\s*(.*?)\s*임차인:\s*(안심|주의)\s*이유:\s*(.*?)(?=\n## 특약|\Z)'
            matches = re.findall(pattern, llm_output, re.MULTILINE | re.DOTALL)
            
            logger.debug(f"정규식 매치 결과: {len(matches)}개")
            
            for match in matches:
                clause_num_str, owner_grade, owner_reason, tenant_grade, tenant_reason = match
                clause_id = int(clause_num_str)  # 실제 특약 ID
                
                logger.debug(f"처리 중: 특약 {clause_id}번")
                
                # 실제 특약 ID로 클로즈 찾기
                if clause_id in clause_id_map:
                    clause = clause_id_map[clause_id]
                    
                    # 등급 파싱
                    owner_assessment = AssessmentLevel.CAUTION if "주의" in owner_grade.strip() else AssessmentLevel.SAFE
                    tenant_assessment = AssessmentLevel.CAUTION if "주의" in tenant_grade.strip() else AssessmentLevel.SAFE
                    
                    # 이유 정리 (줄바꿈을 공백으로 변환)
                    owner_reason_clean = owner_reason.strip().replace('\n', ' ')
                    tenant_reason_clean = tenant_reason.strip().replace('\n', ' ')
                    
                    assessment = ClauseAssessment(
                        clause_id=clause.id,
                        clause_title=clause.title,
                        clause_content=clause.content,
                        owner_assessment=owner_assessment,
                        owner_reason=owner_reason_clean,
                        tenant_assessment=tenant_assessment,
                        tenant_reason=tenant_reason_clean
                    )
                    
                    assessments.append(assessment)
                    logger.debug(f"특약 {clause_id}번 파싱 성공")
                else:
                    logger.warning(f"특약 {clause_id}번을 찾을 수 없음 (가능한 ID: {list(clause_id_map.keys())})")
            
            logger.info(f"배치 파싱 성공: {len(assessments)}개")
            return assessments
            
        except Exception as e:
            logger.error(f"배치 파싱 실패: {e}")
            traceback.print_exc()
            return []

