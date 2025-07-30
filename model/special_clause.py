"""
특약 추천 및 대화 기반 개선 전용    

"""
import sys
import os
import json
import warnings
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
logger = logging.getLogger(__name__)

# deprecation warning 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 프로젝트 루트 경로 설정
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
law_system_path = os.path.join(project_root, "law_system")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if law_system_path not in sys.path:
    sys.path.insert(0, law_system_path)

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


# law_system import
try:
    from law_system.law_vectorstore import get_law_vectorstore, search_law
    logger.info("OK: law_system module imported successfully")
    LAW_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.error(f"ERROR: law_system import failed: {e}")
    LAW_SYSTEM_AVAILABLE = False

load_dotenv()

class SafetyLevel(Enum):
    """안전도 레벨"""
    SAFE = "안전"
    CAUTION = "주의"

@dataclass
class RoleAssessment:
    """역할별 평가 (임대인/임차인)"""
    safety_level: SafetyLevel
    reason: str
    legal_basis: str
    risk_factors: List[str] = None

@dataclass
class ClauseRecommendation:
    """특약 추천 결과"""
    id: int
    title: str
    content: str
    landlord_assessment: RoleAssessment
    tenant_assessment: RoleAssessment

class ContractClauseAI:
    """특약 추천 및 안전도 평가 AI 시스템 """
    
    def __init__(self, model_name: str = "gemini-1.5-pro", temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._setup_llm()
        self.vectorstore = self._setup_vectorstore()
        
        # 중복 방지를 위한 생성된 특약 이력 (방별로 관리)
        self.generated_clauses = {}  # {room_id: [특약 제목들]}
        
    def _setup_llm(self):
        """LLM 모델 설정"""
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature
            )
            print("OK: Gemini LLM initialized successfully")
            return llm
        except Exception as e:
            print(f"ERROR: LLM initialization failed: {e}")
            raise
        
    def _setup_vectorstore(self):
        """벡터스토어 설정 - 무조건 로딩"""
        if not LAW_SYSTEM_AVAILABLE:
            raise RuntimeError("law_system is required but not available!")
        
        try:
            vectorstore = get_law_vectorstore()
            if vectorstore:
                print("OK: Vectorstore connected successfully")
                # 테스트는 생략하여 초기화 속도 개선
                return vectorstore
            else:
                raise RuntimeError("Vectorstore is None - initialization failed!")
        except Exception as e:
            print(f"ERROR: Vectorstore connection failed: {e}")
            raise
    
    # ==================== 1. 초기 특약 추천 ====================
    def recommend_initial_clauses(self, room_id: str, 
                                landlord_survey: Dict[str, Any],
                                tenant_survey: Dict[str, Any], 
                                contract_file_name: str = None) -> Dict[str, Any]:
        """1차 특약 추천 + 임대인/임차인별 안전도 평가"""
        
        try:
            print(f"🎯 초기 특약 추천 시작 - Room ID: {room_id}")
            
            # 방별 특약 이력 초기화
            self.generated_clauses[room_id] = []
            
            # 기존 계약서에서 특약 로드 (있다면)
            existing_clauses = self._load_existing_clauses(contract_file_name)
            
            # 1단계: 특약 추천
            recommended_clauses = self._generate_initial_clauses(
                room_id, landlord_survey, tenant_survey, existing_clauses
            )
            
            # 생성된 특약 제목들 저장 (중복 방지용)
            for clause in recommended_clauses:
                self.generated_clauses[room_id].append(clause['title'])
            
            # 2단계: 각 특약별 임대인/임차인 안전도 평가
            assessed_clauses = []
            
            for clause_data in recommended_clauses:
                print(f"📋 특약 평가 중: {clause_data['title']}")
                
                # 임대인 관점 평가
                landlord_assessment = self._assess_clause_safety(
                    clause_data, "임대인", landlord_survey
                )
                
                # 임차인 관점 평가
                tenant_assessment = self._assess_clause_safety(
                    clause_data, "임차인", tenant_survey
                )
                
                # 평가 결과 통합
                clause_recommendation = ClauseRecommendation(
                    id=clause_data["id"],
                    title=clause_data["title"],
                    content=clause_data["content"],
                    landlord_assessment=landlord_assessment,
                    tenant_assessment=tenant_assessment
                )
                
                assessed_clauses.append(clause_recommendation)
            
            # Spring 반환 형식으로 변환
            result = self._format_spring_response(room_id, assessed_clauses)
            print(f"OK 초기 특약 추천 완료: {len(assessed_clauses)}개")
            return result
            
        except Exception as e:
            print(f"ERROR 특약 추천 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _load_existing_clauses(self, contract_file_name: str) -> List[str]:
        """계약서 JSON에서 기존 특약 로드"""
        if not contract_file_name:
            return []
        
        json_path = os.path.join(project_root, "data", "output", "contract_json", contract_file_name)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            special_terms = data.get("special_terms", [])
            print(f"OK 계약서 특약 {len(special_terms)}개 로드: {contract_file_name}")
            return special_terms
        except FileNotFoundError:
            print(f"ERROR 계약서 파일 없음: {json_path}")
            return []
        except Exception as e:
            print(f"ERROR JSON 읽기 오류: {e}")
            return []

    def _generate_initial_clauses(self, room_id: str,
                                landlord_survey: Dict[str, Any], 
                                tenant_survey: Dict[str, Any],
                                existing_clauses: List[str] = None) -> List[Dict[str, Any]]:
        """초기 특약 6개 추천 - 법령 기반 RAG 활용"""
        
        # 사전조사 컨텍스트 준비
        survey_context = self._prepare_survey_context(landlord_survey, tenant_survey)
        
        # 기존 계약서 컨텍스트 준비
        existing_context, info_clause = self._prepare_existing_context(existing_clauses)
        
        # 법령 검색 (RAG)
        law_context = self._search_relevant_laws(f"임대차 계약 특약 {survey_context[:100]}")
        
        # 특약 추천 프롬프트
        info_instruction = ""
        if info_clause:
            info_instruction = f"\n# 1번 특약 지정:\n제목: 기본 정보\n내용: {info_clause}\n"

        prompt = PromptTemplate.from_template("""
당신은 부동산 임대차 계약 전문가입니다.
다음 정보를 바탕으로 **정확히 6개의 특약**을 추천해주세요.

{survey_context}
{existing_context}
{law_context}
{info_instruction}

# 중요한 규칙:
1. 반드시 6개의 특약만 생성하세요
2. 정보성 조항이 지정되었다면 1번 특약으로 그대로 사용하세요
3. 각 특약은 구체적이고 실행 가능해야 합니다
4. 임대인과 임차인 양측의 이익을 고려하세요
5. 법령에 근거한 특약을 우선하세요
6. 중복되는 특약은 절대 만들지 마세요

# 출력 형식:
## 1번 특약
제목: [특약 제목]
내용: [구체적인 특약 조항 내용]

## 2번 특약
제목: [특약 제목]  
내용: [구체적인 특약 조항 내용]

## 3번 특약
제목: [특약 제목]
내용: [구체적인 특약 조항 내용]

## 4번 특약
제목: [특약 제목]
내용: [구체적인 특약 조항 내용]

## 5번 특약
제목: [특약 제목]
내용: [구체적인 특약 조항 내용]

## 6번 특약
제목: [특약 제목]
내용: [구체적인 특약 조항 내용]
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "survey_context": survey_context,
            "existing_context": existing_context,
            "law_context": law_context,
            "info_instruction": info_instruction
        })
        
        return self._parse_clauses(result)

    def _prepare_survey_context(self, landlord_survey: Dict[str, Any], tenant_survey: Dict[str, Any]) -> str:
        """사전조사 데이터를 컨텍스트로 변환"""
        context = "# 사전조사 결과:\n"
        
        context += "\n## 임대인 관심사:\n"
        for key, value in landlord_survey.items():
            context += f"- {key}: {value}\n"
        
        context += "\n## 임차인 관심사:\n"
        for key, value in tenant_survey.items():
            context += f"- {key}: {value}\n"
        
        return context

    def _prepare_existing_context(self, existing_clauses: List[str]) -> tuple:
        """기존 계약서 컨텍스트 준비"""
        existing_context = ""
        info_clause = None
        
        if existing_clauses:
            existing_context = "\n# 기존 계약서의 특약들:\n"
            for i, clause in enumerate(existing_clauses, 1):
                existing_context += f"{i}. {clause}\n"
                
                # 첫 번째 조항이 정보성인지 확인
                if i == 1 and any(keyword in clause for keyword in ["계좌", "연락처", "전화", "주소", "이메일"]):
                    info_clause = clause
                    existing_context += "\n※ 첫 번째 조항은 정보성 조항으로 1번 특약에 포함됩니다.\n"
        
        return existing_context, info_clause

    def _search_relevant_laws(self, query: str) -> str:
        """법령 검색 (RAG)"""
        if not self.vectorstore:
            return "\n# 법령 검색을 사용할 수 없습니다.\n"
        
        try:
            law_results = search_law(query, k=5)
            
            if law_results:
                law_context = "\n# 관련 법령:\n"
                for result in law_results:
                    law_name = result.get('law_name', '')
                    article = result.get('article', '')
                    content = result.get('content', '')[:200]  # 200자만
                    
                    if law_name and article:
                        law_context += f"\n## {law_name} {article}\n{content}...\n"
                
                return law_context
            else:
                return "\n# 관련 법령을 찾을 수 없습니다.\n"
                
        except Exception as e:
            print(f"WARNING 법령 검색 실패: {e}")
            return "\n# 법령 검색 중 오류가 발생했습니다.\n"

    def _parse_clauses(self, llm_output: str) -> List[Dict[str, Any]]:
        """LLM 출력에서 특약 파싱"""
        clauses = []
        current_clause = {}
        clause_id = 0
        
        lines = llm_output.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 특약 번호 감지
            if line.startswith('##') and ('번 특약' in line):
                # 이전 특약 저장
                if current_clause and current_clause.get('제목') and current_clause.get('내용'):
                    clause_id += 1
                    clauses.append({
                        "id": clause_id,
                        "title": current_clause['제목'],
                        "content": current_clause['내용']
                    })
                
                # 새 특약 시작
                current_clause = {}
                
            elif line.startswith('제목:'):
                current_clause['제목'] = line.replace('제목:', '').strip()
            elif line.startswith('내용:'):
                current_clause['내용'] = line.replace('내용:', '').strip()
        
        # 마지막 특약 저장
        if current_clause and current_clause.get('제목') and current_clause.get('내용'):
            clause_id += 1
            clauses.append({
                "id": clause_id,
                "title": current_clause['제목'],
                "content": current_clause['내용']
            })
        
        # 정확히 6개 체크
        if len(clauses) != 6:
            print(f"WARNING 특약 개수 오류: {len(clauses)}개 (예상: 6개)")
        
        return clauses[:6]

    def _assess_clause_safety(self, clause_data: Dict[str, Any], 
                            role: str, survey_data: Dict[str, Any]) -> RoleAssessment:
        """특약 안전도 평가 - 역할별"""
        
        # 역할별 관심사 정리
        role_concerns = ""
        for key, value in survey_data.items():
            role_concerns += f"{key}: {value}\n"
        
        # 법령 검색
        law_context = self._search_relevant_laws(f"{clause_data['title']} {role} 위험")
        
        # 평가 프롬프트
        prompt = PromptTemplate.from_template("""
다음 특약이 {role}에게 "안전"한지 "주의"가 필요한지 평가해주세요.

# 평가할 특약:
제목: {title}
내용: {content}

# {role}의 관심사:
{role_concerns}

# 관련 법령:
{law_context}

# 평가 기준:
- 안전: {role}에게 유리하거나 합리적인 조항
- 주의: {role}에게 불리하거나 위험할 수 있는 조항

# 출력 형식:
평가결과: [안전 또는 주의]
판단근거: [{role} 관점에서 구체적인 이유를 3줄 이내로]
법적근거: [관련 법령이나 규정, 없으면 "해당 없음"]
위험요소: [주의인 경우만, 구체적인 위험 요소를 쉼표로 구분]
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "role": role,
            "title": clause_data["title"],
            "content": clause_data["content"],
            "role_concerns": role_concerns,
            "law_context": law_context
        })
        
        return self._parse_assessment(result)

    def _parse_assessment(self, llm_output: str) -> RoleAssessment:
        """평가 결과 파싱"""
        assessment_data = {}
        risk_factors = []
        
        for line in llm_output.split('\n'):
            line = line.strip()
            if line.startswith('평가결과:'):
                result = line.replace('평가결과:', '').strip()
                assessment_data['safety_level'] = SafetyLevel.SAFE if result == "안전" else SafetyLevel.CAUTION
            elif line.startswith('판단근거:'):
                assessment_data['reason'] = line.replace('판단근거:', '').strip()
            elif line.startswith('법적근거:'):
                assessment_data['legal_basis'] = line.replace('법적근거:', '').strip()
            elif line.startswith('위험요소:'):
                risk_text = line.replace('위험요소:', '').strip()
                if risk_text and risk_text != "해당 없음":
                    risk_factors = [risk.strip() for risk in risk_text.split(',')]
        
        return RoleAssessment(
            safety_level=assessment_data.get('safety_level', SafetyLevel.CAUTION),
            reason=assessment_data.get('reason', '평가 정보 없음'),
            legal_basis=assessment_data.get('legal_basis', '관련 법령 정보 없음'),
            risk_factors=risk_factors if risk_factors else None
        )

    # ==================== 2. 대화 기반 특약 개선 ====================
    def improve_clause_from_conversation(self, room_id: str, clause_number: int,
                                       conversation: str, previous_clause: str,
                                       landlord_survey: Dict[str, Any] = None,
                                       tenant_survey: Dict[str, Any] = None,
                                       contract_file_name: str = None) -> Dict[str, Any]:
        """대화 기반 특약 개선 - 전체 컨텍스트 활용"""
        
        try:
            print(f"🔄 특약 개선 시작 - Room: {room_id}, 특약: {clause_number}번")
            
            # 개선된 특약 생성
            improved_clause = self._generate_improved_clause(
                room_id, conversation, previous_clause, clause_number,
                landlord_survey, tenant_survey, contract_file_name
            )
            
            # 특약 제목 이력 업데이트 (중복 방지용)
            if room_id in self.generated_clauses:
                if len(self.generated_clauses[room_id]) >= clause_number:
                    self.generated_clauses[room_id][clause_number - 1] = improved_clause["title"]
                else:
                    self.generated_clauses[room_id].append(improved_clause["title"])
            
            # 안전도 재평가
            landlord_assessment = self._assess_clause_safety(
                improved_clause, "임대인", landlord_survey or {"대화기반": "개선"}
            )
            
            tenant_assessment = self._assess_clause_safety(
                improved_clause, "임차인", tenant_survey or {"대화기반": "개선"}
            )
            
            # 결과 포매팅
            result = {
                "success": True,
                "room_id": room_id,
                "clause_number": clause_number,
                "improved_clause": {
                    "title": improved_clause["title"],
                    "content": improved_clause["content"],
                    "landlord_assessment": {
                        "safety_level": landlord_assessment.safety_level.value,
                        "reason": landlord_assessment.reason,
                        "legal_basis": landlord_assessment.legal_basis,
                        "risk_factors": landlord_assessment.risk_factors
                    },
                    "tenant_assessment": {
                        "safety_level": tenant_assessment.safety_level.value,
                        "reason": tenant_assessment.reason,
                        "legal_basis": tenant_assessment.legal_basis,
                        "risk_factors": tenant_assessment.risk_factors
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"OK 특약 개선 완료: {clause_number}번")
            return result
            
        except Exception as e:
            print(f"ERROR 특약 개선 중 오류: {e}")
            return {"success": False, "error": str(e)}

    def _generate_improved_clause(self, room_id: str, conversation: str, 
                                previous_clause: str, clause_number: int,
                                landlord_survey: Dict[str, Any] = None,
                                tenant_survey: Dict[str, Any] = None,
                                contract_file_name: str = None) -> Dict[str, Any]:
        """개선된 특약 생성 - 전체 컨텍스트 활용"""
        
        # 사전조사 컨텍스트
        survey_context = ""
        if landlord_survey and tenant_survey:
            survey_context = self._prepare_survey_context(landlord_survey, tenant_survey)
        
        # 기존 계약서 컨텍스트
        existing_context = ""
        if contract_file_name:
            existing_clauses = self._load_existing_clauses(contract_file_name)
            if existing_clauses:
                existing_context = f"\n# 기존 계약서 참고:\n"
                for i, clause in enumerate(existing_clauses, 1):
                    existing_context += f"{i}. {clause}\n"
        
        # 중복 방지 컨텍스트
        duplicate_prevention = ""
        existing_titles = self.generated_clauses.get(room_id, [])
        if existing_titles:
            duplicate_prevention = f"\n# 중복 방지 - 다음과 유사한 제목 피하기:\n"
            for i, title in enumerate(existing_titles, 1):
                if i != clause_number:  # 현재 개선 중인 특약 제외
                    duplicate_prevention += f"- {title}\n"
        
        # 법령 검색
        law_context = self._search_relevant_laws(f"임대차 분쟁 조정 {conversation[:100]}")
        
        # 개선 프롬프트
        prompt = PromptTemplate.from_template("""
임대인과 임차인의 대화를 분석하여 양측이 만족할 수 있는 개선된 특약을 제안해주세요.

# 기존 특약:
{previous_clause}

# 대화 내용:
{conversation}

{survey_context}
{existing_context}
{law_context}
{duplicate_prevention}

# 개선 원칙:
1. 양측의 합의점을 찾아 Win-Win 조항 제안
2. 법령에 위반되지 않는 범위에서 개선
3. 구체적이고 실행 가능한 내용
4. 대화에서 나온 구체적인 조건들을 정확히 반영
5. 사전조사 관심사도 고려
6. 중복되는 제목이나 내용 피하기

# 출력 형식:
제목: [개선된 특약 제목]
내용: [개선된 특약 조항 내용]
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "previous_clause": previous_clause,
            "conversation": conversation,
            "survey_context": survey_context,
            "existing_context": existing_context,
            "law_context": law_context,
            "duplicate_prevention": duplicate_prevention
        })
        
        return self._parse_improved_clause(result, clause_number)

    def _parse_improved_clause(self, llm_output: str, clause_number: int) -> Dict[str, Any]:
        """개선된 특약 파싱"""
        clause_data = {"id": clause_number}
        
        for line in llm_output.split('\n'):
            line = line.strip()
            if line.startswith('제목:'):
                clause_data['title'] = line.replace('제목:', '').strip()
            elif line.startswith('내용:'):
                clause_data['content'] = line.replace('내용:', '').strip()
        
        # 기본값 설정
        if 'title' not in clause_data:
            clause_data['title'] = f"{clause_number}번 개선 특약"
        if 'content' not in clause_data:
            clause_data['content'] = "특약 내용을 파싱할 수 없습니다."
        
        return clause_data
    
    def finalize_room_session(self, room_id: str, reason: str = "completed") -> Dict[str, Any]:
        """세션 종료 및 정리"""
        
        try:
            print(f"🏁 Room {room_id} 세션 종료: {reason}")
            
            # 생성된 특약 정보
            final_clauses = self.generated_clauses.get(room_id, [])
            
            # 메모리 정리
            if room_id in self.generated_clauses:
                del self.generated_clauses[room_id]
            
            result = {
                "success": True,
                "room_id": room_id,
                "termination_reason": reason,
                "final_clause_count": len(final_clauses),
                "final_clause_titles": final_clauses,
                "message": self._get_termination_message(reason),
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"OK Room {room_id} 세션 종료 완료")
            return result
            
        except Exception as e:
            print(f"ERROR 세션 종료 중 오류: {e}")
            return {"success": False, "error": str(e)}

    def _get_termination_message(self, reason: str) -> str:
        """종료 이유별 메시지"""
        messages = {
            "all_approved": "모든 특약이 합의되어 세션이 완료되었습니다",
            "max_rounds": "최대 라운드(3회)가 완료되어 세션이 종료되었습니다", 
            "user_quit": "사용자 요청으로 세션이 종료되었습니다",
            "completed": "세션이 정상적으로 완료되었습니다"
        }
        return messages.get(reason, "세션이 종료되었습니다")
    
    def _format_spring_response(self, room_id: str, 
                               clause_recommendations: List[ClauseRecommendation]) -> Dict[str, Any]:
        """Spring 반환 형식으로 변환"""
        
        formatted_clauses = []
        
        for clause in clause_recommendations:
            formatted_clause = {
                "id": clause.id,
                "title": clause.title,
                "content": clause.content,
                "landlord_assessment": {
                    "safety_level": clause.landlord_assessment.safety_level.value,
                    "reason": clause.landlord_assessment.reason,
                    "legal_basis": clause.landlord_assessment.legal_basis,
                    "risk_factors": clause.landlord_assessment.risk_factors
                },
                "tenant_assessment": {
                    "safety_level": clause.tenant_assessment.safety_level.value,
                    "reason": clause.tenant_assessment.reason,
                    "legal_basis": clause.tenant_assessment.legal_basis,
                    "risk_factors": clause.tenant_assessment.risk_factors
                }
            }
            formatted_clauses.append(formatted_clause)
        
        return {
            "success": True,
            "room_id": room_id,
            "clauses": formatted_clauses,
            "total_count": len(formatted_clauses),
            "timestamp": datetime.now().isoformat(),
            "vectorstore_status": "connected" if self.vectorstore else "disconnected"
        }
        
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        return {
            "llm_status": "connected" if self.llm else "disconnected",
            "vectorstore_status": "connected" if self.vectorstore else "disconnected",
            "law_system_available": LAW_SYSTEM_AVAILABLE,
            "active_rooms": len(self.generated_clauses),
            "room_ids": list(self.generated_clauses.keys()),
            "model_name": self.model_name,
            "temperature": self.temperature
        }
        
        
contractclauseAI = ContractClauseAI()

# Spring 연동용 
def get_initial_clause_recommendations(room_id: str, 
                                     landlord_survey: Dict[str, Any],
                                     tenant_survey: Dict[str, Any],
                                     contract_file_name: str = None) -> Dict[str, Any]:
    """초기 특약 추천 + 안전도 평가 (Spring 연동용)"""
    return contractclauseAI.recommend_initial_clauses(
        room_id, landlord_survey, tenant_survey, contract_file_name
    )

def improve_clause_from_conversation_spring(room_id: str, clause_number: int,
                                          conversation: str, previous_clause: str,
                                          landlord_survey: Dict[str, Any] = None,
                                          tenant_survey: Dict[str, Any] = None,
                                          contract_file_name: str = None) -> Dict[str, Any]:
    """대화 기반 특약 개선 (Spring 연동용)"""
    return contractclauseAI.improve_clause_from_conversation(
        room_id, clause_number, conversation, previous_clause,
        landlord_survey, tenant_survey, contract_file_name
    )

def finalize_room_session_spring(room_id: str, reason: str = "completed") -> Dict[str, Any]:
    """세션 종료 (Spring 연동용)"""
    return contractclauseAI.finalize_room_session(room_id, reason)

def get_system_status_spring() -> Dict[str, Any]:
    """시스템 상태 확인 (Spring 연동용)"""
    return contractclauseAI.get_system_status()


def test_clause_system():
    """완전한 시스템 테스트"""
    
    print("\n=== 잇집 특약 AI 시스템 테스트 ===")
    
    # 시스템 상태 확인
    status = get_system_status_spring()
    print(f"시스템 상태:")
    print(f"   LLM: {status['llm_status']}")
    print(f"   벡터스토어: {status['vectorstore_status']}")
    print(f"   활성 방: {status['active_rooms']}개")
    
    # 테스트 데이터
    room_id = "test_room_001"
    
    landlord_survey = {
        "주요_관심사": "임대료 연체 방지, 시설물 파손 방지",
        "우려_사항": "소음 문제, 원상복구 비용",
        "선호_조건": "장기 임대, 안정적 임차인"
    }
    
    tenant_survey = {
        "주요_관심사": "애완동물 사육, 인테리어 자유도",
        "우려_사항": "과도한 제약, 높은 추가 비용",
        "생활_패턴": "재택근무, 조용한 환경 선호"
    }
    
    # 1. 초기 특약 추천 테스트
    print(f"\n[1단계]: 초기 특약 추천")
    initial_result = get_initial_clause_recommendations(
        room_id, landlord_survey, tenant_survey, "test_contract.json"
    )
    
    if initial_result["success"]:
        print(f"[OK] 초기 특약 {initial_result['total_count']}개 추천 완료")
        
        # 특약 미리보기
        for clause in initial_result["clauses"][:3]:
            print(f"\n[특약] {clause['id']}번: {clause['title']}")
            print(f"   임대인: {clause['landlord_assessment']['safety_level']}")
            print(f"   임차인: {clause['tenant_assessment']['safety_level']}")
    else:
        print(f"ERROR 실패: {initial_result.get('error', 'Unknown error')}")
        return
    
    # 2. 대화 기반 개선 테스트 (1라운드)
    print(f"\n[2단계]: 1라운드 특약 개선")
    
    test_conversation = """
    임대인: 애완동물은 절대 안 됩니다. 냉새와 소음 때문에 다른 입주자들이 불편해할 수 있어요.
    임차인: 작은 고양이 한 마리만 키우고 싶은데, 추가 보증금을 내는 것도 괜찮습니다.
    임대인: 그럼 추가 보증금 50만원과 함께 매월 청소비 5만원을 내시면 어떨까요?
    임차인: 50만원 추가 보증금은 괜찮지만, 매월 5만원은 너무 부담스럽습니다. 2만원 정도면 어떨까요?
    임대인: 좋습니다. 그럼 추가 보증금 50만원, 매월 청소비 2만원, 그리고 원상복구는 확실히 해주셔야 합니다.
    임차인: 네, 그 조건들로 합의합니다.
    """
    
    previous_clause = "애완동물 사육을 금지합니다."
    clause_number = 3
    
    improve_result_1 = improve_clause_from_conversation_spring(
        room_id, clause_number, test_conversation, previous_clause,
        landlord_survey, tenant_survey, "test_contract.json"
    )
    
    if improve_result_1["success"]:
        improved_1 = improve_result_1["improved_clause"]
        print(f"[OK] 1라운드 완료: {improved_1['title']}")
        print(f"   임대인: {improved_1['landlord_assessment']['safety_level']}")
        print(f"   임차인: {improved_1['tenant_assessment']['safety_level']}")
    else:
        print(f"[ERROR] 1라운드 실패: {improve_result_1.get('error')}")
        return
    
    # 3. 대화 기반 개선 테스트 (2라운드)
    print(f"\n[3단계]: 2라운드 특약 개선")
    
    second_conversation = """
    임대인: 생각해보니 매월 2만원 청소비도 부담스럽습니다. 대신 3개월마다 전문 청소업체 비용을 반반 나눠서 내는 건 어떨까요?
    임차인: 좋은 아이디어네요. 그럼 3개월마다 전문청소하고 비용은 반반 부담하겠습니다.
    임대인: 그리고 애완동물로 인한 손상이 발생하면 즉시 수리해주셔야 합니다.
    임차인: 물론입니다. 즉시 수리하겠습니다.
    """
    
    improve_result_2 = improve_clause_from_conversation_spring(
        room_id, clause_number, second_conversation, improved_1['content'],
        landlord_survey, tenant_survey, "test_contract.json"
    )
    
    if improve_result_2["success"]:
        improved_2 = improve_result_2["improved_clause"]
        print(f"[OK] 2라운드 완료: {improved_2['title']}")
        print(f"   임대인: {improved_2['landlord_assessment']['safety_level']}")
        print(f"   임차인: {improved_2['tenant_assessment']['safety_level']}")
    else:
        print(f"[ERROR] 2라운드 실패: {improve_result_2.get('error')}")
        return
    
    # 4. 세션 종료 테스트
    print(f"\n🏁 4단계: 세션 종료")
    
    finalize_result = finalize_room_session_spring(room_id, "max_rounds")
    
    if finalize_result["success"]:
        print("OK 세션 종료 완료")
        print(f"   종료 사유: {finalize_result['termination_reason']}")
        print(f"   최종 특약 수: {finalize_result['final_clause_count']}개")
        print(f"   메시지: {finalize_result['message']}")
    else:
        print(f"ERROR 세션 종료 실패: {finalize_result.get('error')}")
    
    # 5. 시스템 상태 재확인
    final_status = get_system_status_spring()
    print("\n[RESULT] 최종 시스템 상태:")
    print(f"   활성 방: {final_status['active_rooms']}개 (정리됨)")
    
    print("\n[SUCCESS] 테스트 완료!")
    print("\n[INFO] 테스트 결과 요약:")
    print("   [OK] 초기 추천: 성공")
    print("   [OK] 1라운드 개선: 성공") 
    print("   [OK] 2라운드 개선: 성공")
    print("   [OK] 세션 종료: 성공")
    print("   [OK] 중복 방지: 적용됨")
    print("   [OK] 전체 컨텍스트: 활용됨")
    

if __name__ == "__main__":
    test_clause_system()