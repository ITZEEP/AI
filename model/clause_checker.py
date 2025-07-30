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

load_dotenv()
from config.logger_config import get_logger
logger = get_logger(__name__)

# law_system import
try:
    from law_system.law_vectorstore import get_law_vectorstore, search_law
    LAW_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ law_system import 실패: {e}")
    LAW_SYSTEM_AVAILABLE = False




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
    
    def __init__(self, model_name: str = "gemini-2.5-pro", temperature: float = 0.07):
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
        period_days = (contract_info.contract_expire_date - contract_info.contract_date).days + 1
        period_years = period_days / 365
        contract_parts.append(f"계약기간: {contract_info.contract_date.strftime('%Y년 %m월 %d일')} ~ {contract_info.contract_expire_date.strftime('%Y년 %m월 %d일')} (총 {period_days}일, 약 {period_years:.1f}년)")
        
        # 3. 🔥 임대차 유형 및 금액 정보 (전세/월세 명확히 구분)
        contract_parts.append("=== 임대차 유형 및 금액 정보 ===")
        
        # 전세/월세 판단 로직
        if contract_info.monthly_rent and contract_info.monthly_rent > 0:
            # 월세 계약
            contract_parts.append("🏠 임대차 유형: 월세 계약")
            contract_parts.append(f"보증금: {contract_info.deposit_price:,}원")
            contract_parts.append(f"월세: {contract_info.monthly_rent:,}원")
            
            # 월세의 경우 보증금 비율 계산
            if contract_info.deposit_price and contract_info.monthly_rent:
                monthly_to_deposit_ratio = (contract_info.monthly_rent * 12) / contract_info.deposit_price * 100
                contract_parts.append(f"연간 월세/보증금 비율: {monthly_to_deposit_ratio:.1f}%")
                
        else:
            # 전세 계약  
            contract_parts.append("🏠 임대차 유형: 전세 계약")
            contract_parts.append(f"전세보증금: {contract_info.deposit_price:,}원")
            
            # 전세금 수준 분석 (소액임차인 기준 등)
            if contract_info.deposit_price:
                if contract_info.deposit_price <= 165000000:  # 1억 6천 5백만원 이하
                    contract_parts.append("💡 소액임차인 보호 대상 (더 안전한 보호 받음)")
                else:
                    contract_parts.append("💡 고액 전세 (임대인 재정상태 더 중요)")
        
        # 관리비
        if contract_info.maintenance_fee:
            contract_parts.append(f"관리비: {contract_info.maintenance_fee:,}원")
        
        # 4. 특약사항
        if contract_info.special_clauses:
            contract_parts.append("=== 특약사항 ===")
            for i, clause in enumerate(contract_info.special_clauses, 1):
                contract_parts.append(f"{i}. {clause}")
        
        return "\n".join(contract_parts)
    
    def _analyze_full_contract_with_llm(self, full_contract_text: str, contract_info: ContractInfo) -> List[LegalViolation]:
        """LLM으로 전체 계약서 종합 분석 - 개선된 프롬프트"""
        
        # 전세/월세 여부 판단
        is_jeonse = not (contract_info.monthly_rent and contract_info.monthly_rent > 0)
        
        # 관련 법령 검색 (전체 계약서 내용 기반)
        relevant_laws = self._search_relevant_laws_for_full_contract(full_contract_text, is_jeonse)
        laws_context = self._format_laws_context(relevant_laws)
        
            # 🔥 전세/월세 특성을 반영한 프롬프트
        contract_type_guidance = ""
        
        if is_jeonse:
                contract_type_guidance = """
        ## 🏠 전세 계약 특별 검토사항:
        - 전세보증금 반환 관련 조항의 적정성
        - 임대인의 재정 상태나 근저당권 설정 현황 고려 필요성
        - 전세금 대비 과도한 관리비나 기타 비용 부과 여부
        - 전세 계약임에도 월세 성격의 비용 부과 조항 검토
        - 전세보증금 반환보증보험 가입 관련 조항
        """
        else:
                contract_type_guidance = """
        ## 🏠 월세 계약 특별 검토사항:
        - 월세 연체시 조치 절차의 적정성
        - 월세 인상률의 법정 한도 준수 여부
        - 보증금과 월세 비율의 합리성
        - 관리비 별도 부과의 적정성
        - 월세 선납 조건의 합리성
        """
        
        # 🔥 개선된 전체 계약서 분석 프롬프트
        prompt = PromptTemplate.from_template("""
    당신은 부동산 임대차 법률 전문가입니다. 다음 **전체 계약서**를 종합적으로 검토해주세요.

## 전체 계약서:
{full_contract}

## 관련 법령 정보:
{laws_context}

{contract_type_guidance}

## ⚠️ 중요한 검토 원칙:
1. **실제로 법령에 위반되거나 명백히 불공정한 조항만** 지적해주세요
2. 일반적이고 표준적인 조항은 문제로 지적하지 마세요
3. **당신의 법률 전문 지식**을 바탕으로 각 조항이 관련 법령에 위반되는지 판단해주세요
4. **절차적 정당성**과 **실체적 공정성**을 모두 고려해주세요
5. **계약 해지나 퇴거 관련 조항**에서는 적정한 절차와 기간이 보장되는지도 확인해주세요
6. **극단적인 표현**이 포함된 해지/퇴거 조항은 반드시 확인해주세요
7. **전세/월세 계약의 특성**에 맞는 법령 적용 여부를 확인해주세요

## 검토 범위:
**임대차 계약 전반에 걸쳐 법령 위반이나 불공정한 내용이 있는지 종합적으로 검토**해주세요.

**실제 문제가 발견된 경우에만** 다음 형식으로 출력해주세요:

---위반사항---
위반법령: [구체적인 법령명 (조항 번호 제외)]
위반내용: [문제점을 1줄로 간단히]
내용설명: [왜 문제인지 1-2줄로]
법적근거: [정확한 조항 번호]
개선방안: [실제 계약서에 바로 넣을 수 있는 수정된 조항 내용]
해당조항: [문제가 되는 원본 내용]
---위반사항 끝---

## ✅ 문제가 없는 경우:
"검토 결과 법령에 위반되는 조항이 발견되지 않았습니다."라고 답변해주세요.

## 📝 개선방안 작성 가이드:
- "~~한다" 형식의 완전한 조항으로 작성
- 기존 조항을 법령에 맞게 수정한 완성형 문장
- 바로 계약서에 복사해서 넣을 수 있는 형태
- 예시: "임차인은 계약 해지 시 통상적인 사용으로 인한 마모는 원상복구 의무에서 제외되며, 고의 또는 중과실로 인한 손상에 대해서만 원상복구 의무를 진다."
            """)
        
        chain = prompt | self.llm | StrOutputParser()
    
        try:
            result = chain.invoke({
                "full_contract": full_contract_text,
                "laws_context": laws_context,
                "contract_type_guidance": contract_type_guidance
            })
            
            return self._parse_full_contract_analysis_improved(result)
            
        except Exception as e:
            logger.error(f"❌ 전체 계약서 LLM 분석 실패: {e}")
            return []
    
    def _search_relevant_laws_for_full_contract(self, full_contract_text: str, is_jeonse: bool = True) -> List[Dict[str, Any]]:
        """전체 계약서 기반 관련 법령 검색"""
        if not LAW_SYSTEM_AVAILABLE:
            return []
        
        try:
            # 기본 키워드
            keywords = ["주택임대차보호법", "임대차계약"]
            
            # 전세/월세에 따른 특화 키워드
            if is_jeonse:
                keywords.extend([
                    "전세", "전세보증금", "보증금반환", "소액임차인", 
                    "우선변제권", "대항력", "전세권", "보증보험"
                ])
            else:
                keywords.extend([
                    "월세", "차임", "차임증액", "연체", "월세인상", 
                    "보증금", "관리비"
                ])
            
            # 계약서 내용 기반 추가 키워드
            if "원상복구" in full_contract_text:
                keywords.append("원상복구")
            if "해지" in full_contract_text:
                keywords.extend(["해지", "계약해지"])
            if "전대" in full_contract_text or "양도" in full_contract_text:
                keywords.extend(["전대", "양도"])
            
            search_query = " ".join(list(set(keywords))[:10])
            logger.info(f"🔍 {'전세' if is_jeonse else '월세'} 계약 법령 검색: {search_query}")
            
            return search_law(search_query, k=10)
            
        except Exception as e:
            logger.error(f"❌ 전체 계약서 법령 검색 실패: {e}")
            return []
    
    def _parse_full_contract_analysis_improved(self, llm_result: str) -> List[LegalViolation]:
        """개선된 전체 계약서 LLM 분석 결과 파싱"""
        try:
            logger.info("🔍 전체 계약서 분석 결과 파싱 시작")
            
            violations = []
            
            # "문제없음" 또는 "위반되는 조항이 발견되지 않았습니다" 체크
            if ("문제없음" in llm_result or 
                "위반되는 조항이 발견되지 않았습니다" in llm_result or
                "문제가 없" in llm_result or
                "위반되지 않았습니다" in llm_result):
                logger.info("✅ LLM 판단: 계약서에 법령 위반사항 없음")
                return []
            
            # "---위반사항---"으로 구분된 각 위반사항 파싱
            violation_blocks = llm_result.split("---위반사항---")[1:]  # 첫 번째는 빈 문자열
            
            for block in violation_blocks:
                if "---위반사항 끝---" in block:
                    violation_content = block.split("---위반사항 끝---")[0].strip()
                    violation = self._parse_single_violation_block_improved(violation_content)
                    if violation:
                        violations.append(violation)
            
            logger.info(f"✅ 파싱 완료: {len(violations)}건의 실제 위반사항 발견")
            return violations
            
        except Exception as e:
            logger.error(f"❌ 전체 계약서 분석 결과 파싱 실패: {e}")
            return []
    
    def _parse_single_violation_block_improved(self, violation_content: str) -> Optional[LegalViolation]:
        """개선된 개별 위반사항 블록 파싱"""
        try:
            result_data = {}
            
            for line in violation_content.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == "위반법령":
                        result_data['law_name'] = value
                    elif key == "위반내용":
                        result_data['violation_content'] = value
                    elif key == "내용설명":
                        result_data['explanation'] = value
                    elif key == "법적근거":
                        result_data['legal_basis'] = value
                    elif key == "개선방안":
                        result_data['improvement_example'] = value
                    elif key == "해당조항":
                        result_data['original_clause'] = value
            
            # 필수 정보가 없으면 None 반환
            if not result_data.get('law_name') or not result_data.get('improvement_example'):
                logger.warning("⚠️ 필수 정보 누락으로 위반사항 제외")
                return None
            
            return LegalViolation(
                violation_type=ViolationType.ILLEGAL,  # 실제 위반사항만 파싱하므로 모두 ILLEGAL
                law_name=result_data.get('law_name', '주택임대차보호법'),
                violation_content=result_data.get('violation_content', '법령 위반'),
                explanation=result_data.get('explanation', '해당 조항이 법령에 위반됩니다.'),
                legal_basis=result_data.get('legal_basis', '관련 법령 조항'),
                improvement_example=result_data.get('improvement_example', '전문가 상담 필요'),
                original_clause=result_data.get('original_clause', '해당 조항')
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



    
# import time
# from datetime import datetime

# def test_improved_contract_checker():
#     """개선된 계약서 검토 시스템 테스트 - 시간 측정 포함"""
    
#     # ⏰ 시작 시간 기록
#     start_time = time.time()
#     start_datetime = datetime.now()
    
#     print("\n=== 개선된 계약서 법령 검토 시스템 테스트 ===")
#     print(f"🕐 테스트 시작 시각: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
#     print("=" * 60)
    
#     # 🚨 실제 문제가 있는 특약들로 테스트
#     problematic_clauses = [
#         "임차인은 계약 해지 시 원상복구 비용을 전액 부담한다.",
#             "애완동물 사육을 허가하되, 추가 보증금 50만원을 납부한다.",
#             "임대인은 언제든지 3일 전 통보로 계약을 해지할 수 있다.",  # 명백한 위반
#             "임차인은 전대 및 양도를 할 수 없다."
#     ]
    
#     # ✅ 문제없는 일반 조항들도 포함
#     normal_clauses = [
#         "임차인은 임대차 목적물을 선량한 관리자의 주의로 사용해야 한다.",  # 일반적 조항
#         "월세는 매월 말일까지 지급한다.",  # 표준 조항
#         "계약기간은 2년으로 한다."  # 법정 기간
#     ]
    
#     # 🏗️ 테스트 계약 정보 생성
#     setup_start = time.time()
#     test_contract_info = ContractInfo(
#         contract_id=1,
#         home_id=1,
#         owner_id=1,
#         buyer_id=2,
#         contract_date=datetime(2024, 1, 1),
#         contract_expire_date=datetime(2025, 12, 31),  # 2년 계약
#         deposit_price=300000000,  # 3억원
#         monthly_rent=0,  # 전세
#         maintenance_fee=150000,
#         special_clauses=problematic_clauses + normal_clauses  # 문제 조항 + 일반 조항
#     )
#     setup_time = time.time() - setup_start
#     print(f"📋 계약 정보 생성 시간: {setup_time:.2f}초")
    
#     # 🤖 LLM 초기화 시간 측정
#     print("🔧 AI 모델 초기화 중...")
#     init_start = time.time()
#     checker = get_contract_legal_checker()
#     init_time = time.time() - init_start
#     print(f"🤖 AI 모델 초기화 시간: {init_time:.2f}초")
    
#     # 📊 실제 검토 시간 측정
#     print("🔍 계약서 법령 검토 시작...")
#     analysis_start = time.time()
#     violations = checker.check_contract_legality(test_contract_info)
#     analysis_time = time.time() - analysis_start
#     print(f"⚖️ 법령 검토 분석 시간: {analysis_time:.2f}초")
    
#     # 📈 결과 출력
#     result_start = time.time()
#     if violations:
#         print(f"\n⚠️ 총 {len(violations)}건의 실제 법령 위반사항 발견:")
        
#         # 🎯 "즉시 퇴거" 조항 검출 확인
#         instant_eviction_found = False
        
#         for i, violation in enumerate(violations, 1):
#             print(f"\n--- {i}번째 위반사항 ---")
#             print(f"위반법령: {violation.law_name}")
#             print(f"위반내용: {violation.violation_content}")
#             print(f"내용설명: {violation.explanation}")
#             print(f"법적근거: {violation.legal_basis}")
#             print(f"🔧 개선방안: {violation.improvement_example}")
#             print(f"원본조항: {violation.original_clause}")
            
#             # "즉시 퇴거" 검출 확인
#             if "즉시" in violation.original_clause and "퇴거" in violation.original_clause:
#                 instant_eviction_found = True
#                 print("🎯 *** 즉시 퇴거 조항 검출 성공! ***")
        
#         print(f"\n📊 검출 결과 분석:")
#         print(f"   - 총 위반사항: {len(violations)}건")
#         print(f"   - 즉시 퇴거 조항: {'✅ 검출됨' if instant_eviction_found else '❌ 미검출'}")
        
#     else:
#         print("\n✅ 검토 결과 법령에 위반되는 조항이 발견되지 않았습니다.")
#         print("⚠️ 주의: 명백한 위반 조항들이 있는데 검출되지 않았습니다.")
    
#     result_time = time.time() - result_start
#     print(f"📄 결과 출력 시간: {result_time:.2f}초")
    
#     # ⏰ 총 실행 시간 계산
#     total_time = time.time() - start_time
#     end_datetime = datetime.now()
    
#     print("\n" + "=" * 60)
#     print("⏱️ 시간 측정 결과:")
#     print("=" * 60)
#     print(f"🕐 시작 시각: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"🕕 종료 시각: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"⏱️ 총 실행 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
#     print()
#     print("📊 세부 시간 분석:")
#     print(f"   📋 계약 정보 생성: {setup_time:.2f}초 ({setup_time/total_time*100:.1f}%)")
#     print(f"   🤖 AI 모델 초기화: {init_time:.2f}초 ({init_time/total_time*100:.1f}%)")
#     print(f"   ⚖️ 법령 검토 분석: {analysis_time:.2f}초 ({analysis_time/total_time*100:.1f}%)")
#     print(f"   📄 결과 출력: {result_time:.2f}초 ({result_time/total_time*100:.1f}%)")
#     print()
    
#     # 💡 성능 평가
#     if total_time < 30:
#         performance = "🚀 매우 빠름"
#     elif total_time < 60:
#         performance = "⚡ 빠름"
#     elif total_time < 120:
#         performance = "🐌 보통"
#     else:
#         performance = "🐢 느림"
    
#     print(f"🎯 성능 평가: {performance}")
#     print(f"💡 분석 속도: 특약 {len(problematic_clauses + normal_clauses)}개를 {analysis_time:.1f}초에 처리")


# if __name__ == "__main__":
#     # 기존 테스트
#     test_improved_contract_checker()
    
