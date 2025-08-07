"""
model/clause_checker.py - 계약서 적법성 검사 모델 (원래 프롬프트 완전 적용)

역할:
1. 계약서 전체 텍스트를 입력받아 정교한 적법성 검토
2. 전세/월세별 특화된 법령 위반 조항 식별  
3. 원래 코드의 세밀한 프롬프트와 검토 로직 사용
4. RetrievalQA + law_vectorstore 활용
"""

import sys
import os
from typing import List, Dict, Any, Optional
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


class ContractLegalChecker:
    """계약서 적법성 검사 AI 시스템 (원래 프롬프트 완전 적용)"""
    
    def __init__(self, model_name: str = "gemini-2.5-pro", temperature: float = 0.1):
        """
        Args:
            model_name: 사용할 LLM 모델명
            temperature: LLM temperature (정확성을 위해 낮게 설정)  
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._setup_llm()
        self.vectorstore = self._setup_vectorstore()
        
    def _setup_llm(self):
        """Gemini LLM 설정"""
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature
            )
            logger.info("Gemini LLM 초기화 성공")
            return llm
        except Exception as e:
            logger.error(f"LLM 초기화 실패: {e}")
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
        
    def analyze_contract_text(self, contract_text: str, is_jeonse: bool = True) -> List[Dict[str, Any]]:
        """
        계약서 전체 텍스트 적법성 검사 (원래 코드의 정교한 로직)
        
        Args:
            contract_text: 계약서 전체 텍스트
            is_jeonse: 전세 여부 (True: 전세, False: 월세)
            
        Returns:
            List[Dict]: 위반사항 딕셔너리 리스트
        """
        try:
            logger.info("계약서 텍스트 적법성 검사 시작 (정교한 프롬프트)")
            
            # 1. 관련 법령 검색 (원래 코드의 정교한 로직)
            relevant_laws = self._search_relevant_laws_for_full_contract(contract_text, is_jeonse)
            laws_context = self._format_laws_context(relevant_laws)
            
            # 2. LLM으로 전체 계약서 종합 분석 (원래 프롬프트)
            violations = self._analyze_full_contract_with_llm(contract_text, is_jeonse, laws_context)
            
            logger.info(f"적법성 검사 완료 - 총 {len(violations)}건 문제 발견")
            return violations
            
        except Exception as e:
            logger.error(f"계약서 검사 중 오류: {e}")
            return []
    
    def _search_relevant_laws_for_full_contract(self, contract_text: str, is_jeonse: bool) -> List[Dict[str, Any]]:
        """전체 계약서 기반 관련 법령 검색 (원래 코드 완전 복원)"""
        if not self.vectorstore:
            return []
        
        try:
            all_laws = []
            
            # 🆕 1. 핵심 임대차 법령들을 카테고리별로 포괄적 검색
            comprehensive_queries = [
                # 계약 체결 관련
                "임대차계약 체결 당사자 권리의무",
                "계약서 작성 필수기재사항 특약",
                
                # 계약 해지 관련  
                "계약해지 정당한사유 절차 통지",
                "중도해지 위약금 손해배상",
                
                # 보증금/차임 관련
                "보증금 반환 우선변제권 대항력",
                "차임 연체 인상 한도 제한",
                
                # 임대차 기간/갱신 관련
                "임대차기간 갱신 요구 거절",
                "계약갱신 차임증액 제한",
                
                # 목적물 관리 관련
                "원상복구 의무 자연마모 수선",
                "시설물 하자 수리 의무",
                
                # 권리 양도/전대 관련
                "임차권 양도 전대 동의 제한",
                
                # 특수 상황 관련
                "경매 우선변제 소액임차인 보호",
                "전세보증보험 HUG 보증",
                
                # 분쟁 해결 관련
                "임대차분쟁조정위원회 조정 중재"
            ]
            
            for query in comprehensive_queries:
                laws = search_law(query, k=8)  # 각 카테고리마다 8개씩
                all_laws.extend(laws)
                logger.info(f"포괄적 검색 '{query}': {len(laws)}개 법령 수집")
            
            # 🆕 2. 전세/월세 특화 법령
            if is_jeonse:
                specific_queries = [
                    "전세 전세보증금 전세권 설정",
                    "소액임차인 보호 우선변제",
                    "전세보증보험 가입 혜택"
                ]
            else:
                specific_queries = [
                    "월세 차임 관리비 연체료",
                    "월세인상 한도 통지의무",
                    "보증금 월세 전환비율"
                ]
                
            for query in specific_queries:
                laws = search_law(query, k=8)
                all_laws.extend(laws)
            
            # 🆕 3. 계약서 특정 내용 검출 시 해당 법령 추가 검색
            contract_lower = contract_text.lower()
            specific_queries = []
            
            if "해지" in contract_lower:
                specific_queries.extend([
                    "해지통지 기간 절차",
                    "해지사유 정당성 판단기준"
                ])
            if "원상복구" in contract_lower:
                specific_queries.extend([
                    "원상복구범위 자연마모 제외",
                    "수선의무 임대인 임차인 구분"
                ])
            if "위약금" in contract_lower:
                specific_queries.extend([
                    "위약금 제한 약관규제법",
                    "손해배상 예정 과다금지"
                ])
            if "전대" in contract_lower or "양도" in contract_lower:
                specific_queries.extend([
                    "임차권양도 전대 동의권",
                    "무단전대 계약해지 사유"
                ])
            if "보증보험" in contract_lower:
                specific_queries.append("임대차보증보험 가입조건 혜택")
            if "근저당" in contract_lower:
                specific_queries.append("근저당권 우선변제 경합관계")
            
            for query in specific_queries:
                laws = search_law(query, k=4)
                all_laws.extend(laws)
                logger.info(f"특정 내용 검색 '{query}': {len(laws)}개 법령 수집")
            
            # 🆕 4. 중복 제거 및 법령 다양성 확보
            unique_laws = self._remove_duplicate_laws(all_laws)
            
            # 🆕 5. 법령 소스 다양성 확인
            law_sources = {}
            for law in unique_laws:
                source = law.get('law_name', '기타')
                law_sources[source] = law_sources.get(source, 0) + 1
            
            logger.info(f"수집된 법령 소스별 분포: {law_sources}")
            logger.info(f"총 수집된 법령: {len(all_laws)}개 → 중복 제거 후: {len(unique_laws)}개")
            
            return unique_laws[:80]  # 더 많은 법령 활용 (80개까지)
            
        except Exception as e:
            logger.error(f"포괄적 법령 검색 실패: {e}")
            return []
    
    def _remove_duplicate_laws(self, laws: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 법령 제거"""
        seen_laws = set()
        unique_laws = []
        
        for law in laws:
            # 법령명 + 조항으로 중복 판단
            law_key = f"{law.get('law_name', '')}_{law.get('article', '')}"
            if law_key not in seen_laws and law_key != "_":
                seen_laws.add(law_key)
                unique_laws.append(law)
        
        return unique_laws
    
    def _analyze_full_contract_with_llm(self, full_contract_text: str, is_jeonse: bool, laws_context: str) -> List[Dict[str, Any]]:
        """LLM으로 전체 계약서 종합 분석 - 원래 프롬프트 완전 적용"""
        
        # 전세/월세 특성을 반영한 프롬프트 (원래 코드 완전 복원)
        contract_type_guidance = ""
        
        if is_jeonse:
            contract_type_guidance = """
## 전세 계약 특별 검토사항:
- 전세보증금 반환 관련 조항의 적정성
- 임대인의 재정 상태나 근저당권 설정 현황 고려 필요성
- 전세금 대비 과도한 관리비나 기타 비용 부과 여부
- 전세 계약임에도 월세 성격의 비용 부과 조항 검토
- 전세보증금 반환보증보험 가입 관련 조항
"""
        else:
            contract_type_guidance = """
## 월세 계약 특별 검토사항:
- 월세 연체시 조치 절차의 적정성
- 월세 인상률의 법정 한도 준수 여부
- 보증금과 월세 비율의 합리성
- 관리비 별도 부과의 적정성
- 월세 선납 조건의 합리성
"""
        
        # 개선된 전체 계약서 분석 프롬프트 (원래 코드 완전 복원)
        prompt = PromptTemplate.from_template("""
당신은 부동산 임대차 법률 전문가입니다. 다음 **전체 계약서**를 종합적으로 검토해주세요.

## 전체 계약서:
{full_contract}

## 관련 법령 정보:
{laws_context}

{contract_type_guidance}

## 우리 시스템의 계약 구조 (중요!):
- **전세 계약**: 보증금만 존재 (월세 없음)
- **월세 계약**: 보증금 + 월세 존재
- **계약금은 우리 시스템에 아예 없음** (시퀀스에 포함되지 않음)
- **위약금은 오직 보증금 기준으로만 산정함**

## 중요한 검토 원칙:
1. **실제로 법령에 위반되거나 명백히 불공정한 조항만** 지적해주세요
2. **일반적이고 표준적인 조항은 문제로 지적하지 마세요**(100만원 이하 수선은 임차인 부담)
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

## 문제가 없는 경우:
"검토 결과 법령에 위반되는 조항이 발견되지 않았습니다."라고 답변해주세요.

## 개선방안 작성 가이드:
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
            
            return self._parse_contract_analysis_result(result)
            
        except Exception as e:
            logger.error(f"전체 계약서 LLM 분석 실패: {e}")
            return []
    
    def _format_laws_context(self, laws: List[Dict[str, Any]]) -> str:
        """법령 정보를 컨텍스트로 포맷팅 (원래 코드)"""
        if not laws:
            return "관련 법령 정보가 없습니다."
        
        formatted = []
        for law in laws:
            law_name = law.get('law_name', '법령명 미상')
            article = law.get('article', '')
            content = law.get('content', '')[:400] + "..." if len(law.get('content', '')) > 400 else law.get('content', '')
            
            formatted.append(f"{law_name} {article}\n{content}\n")
        
        return "\n".join(formatted)
    
    def _parse_contract_analysis_result(self, llm_result: str) -> List[Dict[str, Any]]:
        """전체 계약서 LLM 분석 결과 파싱 (원래 코드)"""
        try:
            logger.info("전체 계약서 분석 결과 파싱 시작")
            
            violations = []
            
            # "문제없음" 또는 "위반되는 조항이 발견되지 않았습니다" 체크
            if ("문제없음" in llm_result or 
                "위반되는 조항이 발견되지 않았습니다" in llm_result or
                "문제가 없" in llm_result or
                "위반되지 않았습니다" in llm_result or
                "법령 위반사항" in llm_result and "없" in llm_result):
                logger.info("LLM 판단: 계약서에 법령 위반사항 없음")
                return []
            
            # "---위반사항---"으로 구분된 각 위반사항 파싱
            violation_blocks = llm_result.split("---위반사항---")[1:]  # 첫 번째는 빈 문자열
            
            for block in violation_blocks:
                if "---위반사항 끝---" in block:
                    violation_content = block.split("---위반사항 끝---")[0].strip()
                    violation = self._parse_single_violation_block(violation_content)
                    if violation:
                        violations.append(violation)
            
            logger.info(f"파싱 완료: {len(violations)}건의 실제 위반사항 발견")
            return violations
            
        except Exception as e:
            logger.error(f"전체 계약서 분석 결과 파싱 실패: {e}")
            return []
    
    def _parse_single_violation_block(self, violation_content: str) -> Optional[Dict[str, Any]]:
        """개별 위반사항 블록 파싱 (원래 코드)"""
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
                logger.warning("필수 정보 누락으로 위반사항 제외")
                return None
            
            return {
                'violation_type': ViolationType.ILLEGAL.value,  # 실제 위반사항만 파싱하므로 모두 ILLEGAL
                'law_name': result_data.get('law_name', '주택임대차보호법'),
                'violation_content': result_data.get('violation_content', '법령 위반'),
                'explanation': result_data.get('explanation', '해당 조항이 법령에 위반됩니다.'),
                'legal_basis': result_data.get('legal_basis', '관련 법령 조항'),
                'improvement_example': result_data.get('improvement_example', '전문가 상담 필요'),
                'original_clause': result_data.get('original_clause', '해당 조항')
            }
            
        except Exception as e:
            logger.error(f"개별 위반사항 파싱 실패: {e}")
            return None


# 전역 인스턴스
_contract_checker = None


def get_contract_legal_checker() -> ContractLegalChecker:
    """계약서 검토기 반환 (싱글톤)"""
    global _contract_checker
    
    if _contract_checker is None:
        _contract_checker = ContractLegalChecker()
    
    return _contract_checker


# contract_report.py에서 직접 호출하는 함수
def analyze_contract_text_for_report(contract_text: str, is_jeonse: bool = True) -> List[Dict[str, Any]]:
    """contract_report.py 전용 계약서 분석 함수 (원래 프롬프트 완전 적용)"""
    checker = get_contract_legal_checker()
    return checker.analyze_contract_text(contract_text, is_jeonse)


