"""
model/clause_checker.py - 계약서 법령 적법성 검토 모델

역할:
1. 계약서 조항들을 법령과 대조하여 적법성 검토
2. 위반 조항 식별 및 상세 분석
3. 법적 근거와 개선 방안 제시
4. RetrievalQA + law_vectorstore 활용
"""

import sys
import os
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

# 프로젝트 루트 경로 설정
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
law_system_path = os.path.join(project_root, "law_system")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if law_system_path not in sys.path:
    sys.path.insert(0, law_system_path)

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# law_system import
try:
    from law_system.law_vectorstore import get_law_vectorstore, search_law
    LAW_SYSTEM_AVAILABLE = True
except ImportError as e:
    logging.error(f"❌ law_system import 실패: {e}")
    LAW_SYSTEM_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)


class ViolationType(str, Enum):
    """위반 유형"""
    ILLEGAL = "위반"           # 명백한 법령 위반
    CAUTION = "주의"          # 주의가 필요한 조항
    LEGAL = "적법"            # 법령에 적합


@dataclass
class LegalViolation:
    """법령 위반 정보"""
    violation_type: ViolationType    # 위반 유형
    law_name: str                   # 위반 법령명
    violation_content: str          # 위반 내용
    explanation: str                # 내용 설명
    legal_basis: str                # 법적 근거 (몇 조 몇 항)
    improvement_example: str        # 개선 방안 예시
    original_clause: str            # 원본 조항


@dataclass
class ContractInfo:
    """계약서 정보 (Spring ERD 기반)"""
    contract_id: int
    home_id: int
    owner_id: int
    buyer_id: int
    contract_date: datetime
    contract_expire_date: datetime
    deposit_price: Optional[int] = None
    monthly_rent: Optional[int] = None
    maintenance_fee: Optional[int] = None
    special_clauses: List[str] = None  # 특약사항들


class ContractLegalChecker:
    """계약서 법령 적법성 검토 AI 시스템"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.1):
        """
        Args:
            model_name: 사용할 LLM 모델명
            temperature: LLM temperature (정확성을 위해 낮게 설정)  
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._setup_llm()
        self.vectorstore = self._setup_vectorstore()
        self.retrieval_qa = self._setup_retrieval_qa()
        
    def _setup_llm(self):
        """Gemini LLM 설정"""
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature
            )
            logger.info("✅ Gemini LLM 초기화 성공")
            return llm
        except Exception as e:
            logger.error(f"❌ LLM 초기화 실패: {e}")
            raise
            
    def _setup_vectorstore(self):
        """법령 벡터스토어 설정"""
        if not LAW_SYSTEM_AVAILABLE:
            logger.warning("⚠️ law_system을 사용할 수 없습니다.")
            return None
        
        try:
            vectorstore = get_law_vectorstore()
            if vectorstore:
                logger.info("✅ 법령 벡터스토어 연결 성공")
                return vectorstore
            else:
                logger.warning("❌ 벡터스토어가 None입니다")
                return None
        except Exception as e:
            logger.error(f"❌ 벡터스토어 연결 실패: {e}")
            return None
    
    def _setup_retrieval_qa(self):
        """RetrievalQA 체인 설정"""
        if not self.vectorstore:
            return None
            
        try:
            retriever = self.vectorstore.get_retriever(search_kwargs={"k": 10})
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            logger.info("✅ RetrievalQA 체인 초기화 성공")
            return qa_chain
            
        except Exception as e:
            logger.error(f"❌ RetrievalQA 초기화 실패: {e}")
            return None
    
    def check_contract_legality(self, contract_info: ContractInfo) -> List[LegalViolation]:
        """
        계약서 전체 법령 적법성 검토 - 모든 계약 정보 종합 분석
        
        Args:
            contract_info: 계약서 정보 (Spring ERD 기반)
            
        Returns:
            LLM이 판단한 전체 계약서의 위반 사항 리스트
        """
        try:
            logger.info(f"🔍 전체 계약서 법령 검토 시작 - contract_id: {contract_info.contract_id}")
            
            # 전체 계약서 정보를 하나의 텍스트로 구성
            full_contract_text = self._compose_full_contract_text(contract_info)
            
            # LLM으로 전체 계약서 분석
            violations = self._analyze_full_contract_with_llm(full_contract_text, contract_info)
            
            logger.info(f"✅ 전체 계약서 검토 완료 - 총 {len(violations)}건 문제 발견")
            return violations
            
        except Exception as e:
            logger.error(f"❌ 계약서 검토 중 오류: {e}")
            return []
    
    def _compose_full_contract_text(self, contract_info: ContractInfo) -> str:
        """전체 계약서 정보를 하나의 텍스트로 구성"""
        
        contract_parts = []
        
        # 1. 기본 계약 정보
        contract_parts.append("=== 기본 계약 정보 ===")
        contract_parts.append(f"계약 ID: {contract_info.contract_id}")
        contract_parts.append(f"매물 ID: {contract_info.home_id}")
        
        # 2. 계약 기간
        period_days = (contract_info.contract_expire_date - contract_info.contract_date).days
        contract_parts.append(f"계약기간: {contract_info.contract_date.strftime('%Y년 %m월 %d일')} ~ {contract_info.contract_expire_date.strftime('%Y년 %m월 %d일')} (총 {period_days}일)")
        
        # 3. 금액 정보
        contract_parts.append("=== 금액 정보 ===")
        if contract_info.deposit_price:
            contract_parts.append(f"보증금: {contract_info.deposit_price:,}원")
        if contract_info.monthly_rent:
            contract_parts.append(f"월세: {contract_info.monthly_rent:,}원")
        if contract_info.maintenance_fee:
            contract_parts.append(f"관리비: {contract_info.maintenance_fee:,}원")
        
        # 4. 특약사항
        if contract_info.special_clauses:
            contract_parts.append("=== 특약사항 ===")
            for i, clause in enumerate(contract_info.special_clauses, 1):
                contract_parts.append(f"{i}. {clause}")
        
        return "\n".join(contract_parts)
    
    def _analyze_full_contract_with_llm(self, full_contract_text: str, contract_info: ContractInfo) -> List[LegalViolation]:
        """LLM으로 전체 계약서 종합 분석"""
        
        # 관련 법령 검색 (전체 계약서 내용 기반)
        relevant_laws = self._search_relevant_laws_for_full_contract(full_contract_text)
        laws_context = self._format_laws_context(relevant_laws)
        
        # 전체 계약서 분석 프롬프트
        prompt = PromptTemplate.from_template("""
당신은 부동산 임대차 법률 전문가입니다. 다음 **전체 계약서**를 종합적으로 검토해주세요.

## 전체 계약서:
{full_contract}

## 관련 법령 정보:
{laws_context}

## 검토 범위:
1. 계약기간이 법령에 적합한지 (2년 미만 계약의 경우)
2. 보증금이 소액임차인 보호 범위 내인지
3. 월세/관리비가 합리적인지
4. 특약사항들이 법령에 위반되거나 불공정한지
5. 전체적으로 임차인에게 과도하게 불리한 계약인지

위반사항이 발견될 때마다 다음 형식으로 각각 따로 출력해주세요:

---위반사항---
위반유형: [위반/주의/적법]
위반법령: [구체적인 법령명]
위반내용: [문제점을 1줄로 간단히]
내용설명: [왜 문제인지 1-2문장으로]
법적근거: [조항 번호]
개선방안: [수정 방법을 1문장으로]
해당조항: [문제가 되는 원본 내용]
---위반사항 끝---

## 주의사항:
- 실제로 법령 위반이나 불공정한 내용만 지적해주세요
- 일반적이고 표준적인 조항은 문제로 지적하지 마세요
- 문제가 없다면 "문제없음"이라고 답변해주세요
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({
                "full_contract": full_contract_text,
                "laws_context": laws_context
            })
            
            return self._parse_full_contract_analysis(result)
            
        except Exception as e:
            logger.error(f"❌ 전체 계약서 LLM 분석 실패: {e}")
            return []
    
    def _search_relevant_laws_for_full_contract(self, full_contract_text: str) -> List[Dict[str, Any]]:
        """전체 계약서 기반 관련 법령 검색"""
        if not LAW_SYSTEM_AVAILABLE:
            return []
        
        try:
            # 계약서에서 핵심 키워드 추출
            keywords = []
            
            if "보증금" in full_contract_text:
                keywords.extend(["보증금", "소액임차인", "대항력"])
            if "월세" in full_contract_text:
                keywords.extend(["월세", "차임증액"])
            if "계약기간" in full_contract_text:
                keywords.extend(["계약기간", "존속기간", "갱신"])
            
            # 특약 관련 키워드도 추가
            if "원상복구" in full_contract_text:
                keywords.append("원상복구")
            if "해지" in full_contract_text:
                keywords.extend(["해지", "계약해지"])
            if "전대" in full_contract_text or "양도" in full_contract_text:
                keywords.extend(["전대", "양도"])
            
            # 기본 키워드 추가
            keywords.extend(["주택임대차보호법", "임대차계약"])
            
            search_query = " ".join(list(set(keywords))[:10])  # 중복 제거 후 최대 10개
            logger.info(f"🔍 전체 계약서 법령 검색: {search_query}")
            
            return search_law(search_query, k=10)
            
        except Exception as e:
            logger.error(f"❌ 전체 계약서 법령 검색 실패: {e}")
            return []
    
    def _parse_full_contract_analysis(self, llm_result: str) -> List[LegalViolation]:
        """전체 계약서 LLM 분석 결과 파싱"""
        try:
            logger.info(f"🔍 전체 계약서 분석 결과 파싱 시작")
            
            violations = []
            
            # "문제없음"인 경우 빈 리스트 반환
            if "문제없음" in llm_result or "문제가 없" in llm_result:
                logger.info("✅ LLM 판단: 계약서에 문제없음")
                return []
            
            # "---위반사항---"으로 구분된 각 위반사항 파싱
            violation_blocks = llm_result.split("---위반사항---")[1:]  # 첫 번째는 빈 문자열
            
            for block in violation_blocks:
                if "---위반사항 끝---" in block:
                    violation_content = block.split("---위반사항 끝---")[0].strip()
                    violation = self._parse_single_violation_block(violation_content)
                    if violation:
                        violations.append(violation)
            
            logger.info(f"✅ 파싱 완료: {len(violations)}건의 위반사항 발견")
            return violations
            
        except Exception as e:
            logger.error(f"❌ 전체 계약서 분석 결과 파싱 실패: {e}")
            return []
    
    def _parse_single_violation_block(self, violation_content: str) -> Optional[LegalViolation]:
        """개별 위반사항 블록 파싱"""
        try:
            result_data = {}
            
            for line in violation_content.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == "위반유형":
                        if "위반" in value:
                            result_data['violation_type'] = ViolationType.ILLEGAL
                        elif "주의" in value:
                            result_data['violation_type'] = ViolationType.CAUTION
                        else:
                            result_data['violation_type'] = ViolationType.LEGAL
                    elif key == "위반법령":
                        result_data['law_name'] = value if value != "관련 법령" else "주택임대차보호법"
                    elif key == "위반내용":
                        result_data['violation_content'] = value
                    elif key == "내용설명":
                        result_data['explanation'] = value
                    elif key == "법적근거":
                        result_data['legal_basis'] = value if value != "관련 조항" else "해당 법령 조항"
                    elif key == "개선방안":
                        result_data['improvement_example'] = value
                    elif key == "해당조항":
                        result_data['original_clause'] = value
            
            # 적법한 경우 None 반환
            if result_data.get('violation_type') == ViolationType.LEGAL:
                return None
            
            return LegalViolation(
                violation_type=result_data.get('violation_type', ViolationType.CAUTION),
                law_name=result_data.get('law_name', '주택임대차보호법'),
                violation_content=result_data.get('violation_content', '검토 필요'),
                explanation=result_data.get('explanation', '상세 검토가 필요합니다.'),
                legal_basis=result_data.get('legal_basis', '해당 법령 조항'),
                improvement_example=result_data.get('improvement_example', '전문가 상담 권장'),
                original_clause=result_data.get('original_clause', '해당 없음')
            )
            
        except Exception as e:
            logger.error(f"❌ 개별 위반사항 파싱 실패: {e}")
            return None
    
    def _search_relevant_laws(self, clause: str) -> List[Dict[str, Any]]:
        """조항과 관련된 법령 검색"""
        if not LAW_SYSTEM_AVAILABLE:
            return []
        
        try:
            # 핵심 키워드 추출하여 검색 정확도 향상
            keywords = self._extract_legal_keywords(clause)
            search_query = f"{clause} {' '.join(keywords)}"
            
            return search_law(search_query, k=10)
            
        except Exception as e:
            logger.error(f"❌ 법령 검색 실패: {e}")
            return []
    
    def _extract_legal_keywords(self, clause: str) -> List[str]:
        """조항에서 법적 핵심 키워드 추출"""
        legal_keywords = [
            "임대료", "보증금", "월세", "전세", "관리비",
            "계약기간", "갱신", "해지", "연장", "종료",
            "수리", "보수", "원상복구", "하자", "시설",
            "반려동물", "애완동물", "소음", "흡연",
            "전대", "전차", "양도", "양수", "승계",
            "보증보험", "임대차보호법", "소액임차인"
        ]
        
        found_keywords = [keyword for keyword in legal_keywords if keyword in clause]
        return found_keywords
    
    # def _analyze_with_retrieval_qa(self, clause: str) -> Optional[LegalViolation]:
    #     """RetrievalQA를 사용한 법령 검토 - 개선된 버전"""
        
    #     try:
    #         query = f"""
    #         다음 임대차 계약서 특약 조항이 주택임대차보호법이나 기타 관련 법령에 위반되는지 검토해주세요:
            
    #         특약 조항: {clause}
            
    #         다음 형식으로 정확히 답변해주세요:
    #         위반유형: [위반/주의/적법]
    #         위반법령: [구체적인 법령명]
    #         위반내용: [어떤 부분이 문제인지]
    #         내용설명: [왜 문제가 되는지 상세 설명]
    #         법적근거: [구체적인 조항 예: 제6조 제1항]
    #         개선방안: [수정 방법 제시]
    #         """
            
    #         result = self.retrieval_qa.invoke({"query": query})  # __call__ → invoke로 변경
            
    #         if result and "result" in result:
    #             return self._parse_qa_result_improved(result["result"], clause)
            
    #         return None
            
    #     except Exception as e:
    #         logger.error(f"❌ RetrievalQA 분석 실패: {e}")
    #         return None
    
    # def _parse_qa_result_improved(self, qa_result: str, original_clause: str) -> Optional[LegalViolation]:
    #     """개선된 RetrievalQA 결과 파싱"""
    #     try:
    #         logger.info(f"🔍 QA 결과 파싱: {qa_result[:100]}...")
            
    #         result_data = {}
            
    #         # 줄별로 분석하여 정보 추출
    #         for line in qa_result.split('\n'):
    #             line = line.strip()
    #             if ':' in line:
    #                 key, value = line.split(':', 1)
    #                 key = key.strip()
    #                 value = value.strip()
                    
    #                 if key == "위반유형":
    #                     if "위반" in value:
    #                         result_data['violation_type'] = ViolationType.ILLEGAL
    #                     elif "주의" in value:
    #                         result_data['violation_type'] = ViolationType.CAUTION
    #                     else:
    #                         result_data['violation_type'] = ViolationType.LEGAL
    #                 elif key == "위반법령":
    #                     result_data['law_name'] = value
    #                 elif key == "위반내용":
    #                     result_data['violation_content'] = value
    #                 elif key == "내용설명":
    #                     result_data['explanation'] = value
    #                 elif key == "법적근거":
    #                     result_data['legal_basis'] = value
    #                 elif key == "개선방안":
    #                     result_data['improvement_example'] = value
            
    #         # 형식이 맞지 않으면 직접 LLM 분석으로 대체
    #         if not result_data.get('law_name') or result_data.get('law_name') == '관련 법령':
    #             logger.warning("⚠️ QA 결과 형식 불일치, LLM 직접 분석으로 전환")
    #             return None  # LLM 직접 분석으로 넘어감
            
    #         # 적법한 경우 None 반환
    #         if result_data.get('violation_type') == ViolationType.LEGAL:
    #             return None
            
    #         return LegalViolation(
    #             violation_type=result_data.get('violation_type', ViolationType.CAUTION),
    #             law_name=result_data.get('law_name', '관련 법령'),
    #             violation_content=result_data.get('violation_content', '검토 필요'),
    #             explanation=result_data.get('explanation', '상세 검토가 필요합니다.'),
    #             legal_basis=result_data.get('legal_basis', '관련 조항'),
    #             improvement_example=result_data.get('improvement_example', '전문가 상담 권장'),
    #             original_clause=original_clause
    #         )
            
    #     except Exception as e:
    #         logger.error(f"❌ QA 결과 파싱 실패: {e}")
    #         return None
    
#     def _analyze_clause_with_llm(self, clause: str, relevant_laws: List[Dict[str, Any]]) -> Optional[LegalViolation]:
#         """직접 LLM을 사용한 조항 적법성 분석"""
        
#         # 관련 법령 정보 포맷팅
#         laws_context = self._format_laws_context(relevant_laws)
        
#         prompt = PromptTemplate.from_template("""
# 당신은 부동산 임대차 법률 전문가입니다. 다음 계약서 특약 조항을 검토해주세요.

# ## 검토할 특약 조항:
# {clause}

# ## 관련 법령 정보:
# {laws_context}

# 당신의 전문적 판단으로 이 조항이 법령에 위반되거나 불공정한지 스스로 결정해주세요.
# 일반적이고 표준적인 조항이라면 "적법"으로, 실제로 문제가 있다고 판단되면 "위반" 또는 "주의"로 분류해주세요.

# ## 출력 형식:
# 위반유형: [위반/주의/적법]
# 위반법령: [구체적인 법령명]
# 위반내용: [문제점을 1줄로 간단히]
# 내용설명: [왜 문제인지 1-2문장으로]
# 법적근거: [조항 번호]
# 개선방안: [수정 방법을 1문장으로]
#         """)
        
#         chain = prompt | self.llm | StrOutputParser()
        
#         try:
#             result = chain.invoke({
#                 "clause": clause,
#                 "laws_context": laws_context
#             })
            
#             return self._parse_llm_result(result, clause)
            
#         except Exception as e:
#             logger.error(f"❌ LLM 분석 실패: {e}")
#             return None
    
    # def _check_contract_conditions(self, contract_info: ContractInfo) -> List[LegalViolation]:
    #     """계약 기본 조건들의 법령 적합성 검토 - 순수 LLM 방식"""
    #     violations = []
        
    #     try:
    #         # 계약 조건들을 하나의 텍스트로 구성
    #         contract_conditions = []
            
    #         if contract_info.deposit_price:
    #             contract_conditions.append(f"보증금: {contract_info.deposit_price:,}원")
                
    #         if contract_info.monthly_rent:
    #             contract_conditions.append(f"월세: {contract_info.monthly_rent:,}원")
                
    #         if contract_info.maintenance_fee:
    #             contract_conditions.append(f"관리비: {contract_info.maintenance_fee:,}원")
            
    #         # 계약기간 정보
    #         period_days = (contract_info.contract_expire_date - contract_info.contract_date).days
    #         contract_conditions.append(f"계약기간: {contract_info.contract_date.strftime('%Y년 %m월 %d일')} ~ {contract_info.contract_expire_date.strftime('%Y년 %m월 %d일')} (총 {period_days}일)")
            
    #         # 모든 조건을 하나의 조항으로 취급하여 LLM으로 검토
    #         if contract_conditions:
    #             combined_conditions = " / ".join(contract_conditions)
    #             violation = self._check_clause_legality(combined_conditions)
    #             if violation and violation.violation_type != ViolationType.LEGAL:
    #                 violations.append(violation)
                    
    #     except Exception as e:
    #         logger.error(f"❌ 계약 조건 검토 실패: {e}")
        
    #     return violations
    

    
    def _format_laws_context(self, laws: List[Dict[str, Any]]) -> str:
        """법령 정보를 컨텍스트로 포맷팅"""
        if not laws:
            return "관련 법령 정보가 없습니다."
        
        formatted = []
        for law in laws:
            law_name = law.get('law_name', '법령명 미상')
            article = law.get('article', '')
            content = law.get('content', '')[:300] + "..." if len(law.get('content', '')) > 300 else law.get('content', '')
            
            formatted.append(f"📋 {law_name} {article}\n{content}\n")
        
        return "\n".join(formatted)
    
    # def _parse_qa_result(self, qa_result: str, original_clause: str) -> Optional[LegalViolation]:
    #     """기존 QA 결과 파싱 - 사용 안함"""
    #     return None  # 항상 None 반환하여 LLM 직접 분석으로 넘어가게 함
    
    def _parse_llm_result(self, llm_result: str, original_clause: str) -> Optional[LegalViolation]:
        """개선된 LLM 분석 결과 파싱"""
        try:
            logger.info(f"🔍 LLM 결과 파싱: {llm_result[:100]}...")
            
            result_data = {}
            
            for line in llm_result.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == "위반유형":
                        if "위반" in value:
                            result_data['violation_type'] = ViolationType.ILLEGAL
                        elif "주의" in value:
                            result_data['violation_type'] = ViolationType.CAUTION
                        else:
                            result_data['violation_type'] = ViolationType.LEGAL
                    elif key == "위반법령":
                        result_data['law_name'] = value if value != "관련 법령" else "주택임대차보호법"
                    elif key == "위반내용":
                        result_data['violation_content'] = value
                    elif key == "내용설명":  
                        result_data['explanation'] = value
                    elif key == "법적근거":
                        result_data['legal_basis'] = value if value != "관련 조항" else "해당 법令 조항"
                    elif key == "개선방안":
                        result_data['improvement_example'] = value
            
            # 적법한 경우 None 반환
            if result_data.get('violation_type') == ViolationType.LEGAL:
                return None
            
            # 필수 정보 기본값 설정
            law_name = result_data.get('law_name', '주택임대차보호법')
            legal_basis = result_data.get('legal_basis', '관련 조항')
            
            # "관련 법령" 같은 모호한 표현 개선
            if law_name == "관련 법령":
                law_name = "주택임대차보호법"
            if legal_basis == "관련 조항" or legal_basis == "QA 시스템 분석 결과":
                legal_basis = "해당 법령 조항"
            
            return LegalViolation(
                violation_type=result_data.get('violation_type', ViolationType.CAUTION),
                law_name=law_name,
                violation_content=result_data.get('violation_content', '검토 필요'),
                explanation=result_data.get('explanation', '상세 검토가 필요합니다.'),
                legal_basis=legal_basis,
                improvement_example=result_data.get('improvement_example', '전문가 상담 권장'),
                original_clause=original_clause
            )
            
        except Exception as e:
            logger.error(f"❌ LLM 결과 파싱 실패: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        return {
            "llm_status": "connected" if self.llm else "disconnected",
            "vectorstore_status": "connected" if self.vectorstore else "disconnected", 
            "retrieval_qa_status": "connected" if self.retrieval_qa else "disconnected",
            "law_system_available": LAW_SYSTEM_AVAILABLE,
            "model_name": self.model_name,
            "temperature": self.temperature
        }


# 전역 인스턴스 (싱글톤 패턴)
_contract_checker = None


def get_contract_legal_checker() -> ContractLegalChecker:
    """계약서 법령 검토기 반환"""
    global _contract_checker
    
    if _contract_checker is None:
        _contract_checker = ContractLegalChecker()
    
    return _contract_checker


# Spring 연동용 편의 함수
def check_contract_legality_for_spring(contract_id: int,
                                      home_id: int,
                                      owner_id: int,
                                      buyer_id: int,
                                      contract_date: datetime,
                                      contract_expire_date: datetime,
                                      special_clauses: List[str],
                                      deposit_price: Optional[int] = None,
                                      monthly_rent: Optional[int] = None,
                                      maintenance_fee: Optional[int] = None) -> List[LegalViolation]:
    """Spring용 계약서 법령 검토 편의 함수"""
    
    contract_info = ContractInfo(
        contract_id=contract_id,
        home_id=home_id,
        owner_id=owner_id,
        buyer_id=buyer_id,
        contract_date=contract_date,
        contract_expire_date=contract_expire_date,
        deposit_price=deposit_price,
        monthly_rent=monthly_rent,
        maintenance_fee=maintenance_fee,
        special_clauses=special_clauses
    )
    
    checker = get_contract_legal_checker()
    return checker.check_contract_legality(contract_info)


# 테스트용
if __name__ == "__main__":
    print("\n=== 계약서 법령 검토 시스템 테스트 ===")
    
    # 시스템 상태 확인
    checker = get_contract_legal_checker()
    status = checker.get_system_status()
    print(f"🔧 시스템 상태:")
    print(f"   LLM: {status['llm_status']}")
    print(f"   벡터스토어: {status['vectorstore_status']}")
    print(f"   RetrievalQA: {status['retrieval_qa_status']}")
    
    # 테스트 특약들 - LLM이 스스로 판단하도록
    test_clauses = [
        "임차인은 계약 해지 시 원상복구 비용을 전액 부담한다.",
        "애완동물 사육을 허가하되, 추가 보증금 50만원을 납부한다.", 
        "임대인은 언제든지 3일 전 통보로 계약을 해지할 수 있다.",
        "임차인은 전대 및 양도를 할 수 없다."
    ]
    
    # 테스트 계약 정보
    contract_info = ContractInfo(
        contract_id=1,
        home_id=1,
        owner_id=1,
        buyer_id=2,
        contract_date=datetime(2024, 1, 1),
        contract_expire_date=datetime(2025, 12, 31),
        deposit_price=150000000,  # 1.5억원
        monthly_rent=500000,
        maintenance_fee=100000,
        special_clauses=test_clauses
    )
    
    # 법령 검토 실행
    violations = checker.check_contract_legality(contract_info)
    
    if violations:
        print(f"\n⚠️ 총 {len(violations)}건의 문제점 발견:")
        for i, violation in enumerate(violations, 1):
            print(f"\n--- {i}번째 문제점 ---")
            print(f"위반유형: {violation.violation_type}")
            print(f"위반법령: {violation.law_name}")
            print(f"위반내용: {violation.violation_content}")
            print(f"내용설명: {violation.explanation}")
            print(f"법적근거: {violation.legal_basis}")
            print(f"개선방안: {violation.improvement_example}")
            print(f"원본조항: {violation.original_clause}")
    else:
        print("\n✅ 검토 결과 법령 위반 사항이 발견되지 않았습니다.")
    
    print("\n🎉 테스트 완료!")