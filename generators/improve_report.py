"""
특약 개선을 위한 데이터 파서 및 결과 포맷터
Spring에서 전달받은 대화 데이터를 파싱하고 AI 모델에서 사용할 수 있는 형태로 변환
개선된 특약을 Spring 포맷으로 변환하여 반환
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import sys
import os
from enum import Enum

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.logger_config import get_logger
logger = get_logger(__name__)

try:
    from generators.clause_report import ClauseDataParser
except ImportError as e:
    logger.warning(f"clause_report import failed: {e}")
    ClauseDataParser = None

class AssessmentLevel(Enum):
    SAFE = "안심"
    CAUTION = "주의"

@dataclass
class ClauseHistoryData:
    """이전 특약 히스토리 데이터"""
    title: str
    content: str
    messages: str


@dataclass
class ClauseImprovementRequest:
    """특약 개선 요청 데이터"""
    contract_chat_id: int
    order: int
    round: int
    prev_data: List[ClauseHistoryData]
    recent_data: ClauseHistoryData
    owner_data: Optional[Dict] = None  # 임대인 사전조사 JSON
    tenant_data: Optional[Dict] = None  # 임차인 사전조사 JSON
    ocr_data: Optional[Dict] = None # OCR 결과 JSON
    

@dataclass
class ImprovedClause:
    """개선된 특약 데이터"""
    round: int
    order: int
    title: str
    content: str
    owner_assessment: AssessmentLevel
    owner_reason: str
    tenant_assessment: AssessmentLevel
    tenant_reason: str


class ClauseImprovementParser:
    """특약 개선 요청 파서"""
    
    @staticmethod
    def parse_improvement_request(data: Dict) -> ClauseImprovementRequest:
        """Spring에서 받은 개선 요청 JSON 파싱"""
        try:
            # 이전 데이터 파싱
            prev_data = []
            for prev_item in data.get('prevData', []):
                prev_data.append(ClauseHistoryData(
                    title=prev_item.get('title', ''),
                    content=prev_item.get('content', ''),
                    messages=prev_item.get('messages', '')
                ))
            
            # 최근 데이터 파싱
            recent_data_dict = data.get('recentData', {})
            recent_data = ClauseHistoryData(
                title=recent_data_dict.get('title', ''),
                content=recent_data_dict.get('content', ''),
                messages=recent_data_dict.get('messages', '')
            )
            
            return ClauseImprovementRequest(
                contract_chat_id=data['contractChatId'],
                order=data['order'],
                round=data['round'],
                prev_data=prev_data,
                recent_data=recent_data,
                owner_data=data.get('ownerData'),  # 사전조사 데이터 추가
                tenant_data=data.get('tenantData'),  # 사전조사 데이터 추가
                ocr_data=data.get('ocrData')  # OCR 데이터 추가
            )
            
        except Exception as e:
            logger.error(f"개선 요청 파싱 실패: {e}")
            raise
    
    @staticmethod
    def create_context_for_llm(request: ClauseImprovementRequest) -> str:
        """LLM에서 사용할 컨텍스트 문자열 생성 (라운드별 맞춤 + 사전조사 포함)"""
        context = f"""# 특약 개선 요청 정보:
- 계약 채팅 ID: {request.contract_chat_id}
- 특약 번호: {request.order}번
- 개선 라운드: {request.round}라운드
"""
        # 파서 인스턴스를 한 번만 생성
        parser = ClauseDataParser() if ClauseDataParser is not None else None
        
        
        # 사전조사 정보 추가 (clause_report 모듈 활용)
        if request.owner_data or request.tenant_data:
            context += "\n# 사전조사 정보:\n"
            
            if request.owner_data:
                try:
                    if parser is None:
                        context += "- 임대인 사전조사 정보: 파서 모듈 없음\n"
                    else:
                        parser = ClauseDataParser()
                        owner = parser.parse_owner_precheck(request.owner_data)
                        owner_context = parser.create_context_from_owner(owner)
                        context += owner_context + "\n"
                except Exception as e:
                    logger.warning(f"임대인 사전조사 파싱 실패: {e}")
                    context += "- 임대인 사전조사 정보: 파싱 실패\n"
            
            if request.tenant_data:
                try:
                    if parser is None:
                        context += "- 임차인 사전조사 정보: 파서 모듈 없음\n"
                    else:
                        parser = ClauseDataParser()
                        tenant = parser.parse_tenant_precheck(request.tenant_data)
                        tenant_context = parser.create_context_from_tenant(tenant)
                        context += tenant_context + "\n"
                except Exception as e:
                    logger.warning(f"임차인 사전조사 파싱 실패: {e}")
                    context += "- 임차인 사전조사 정보: 파싱 실패\n"
            
            # OCR 정보 추가
            if request.ocr_data:
                try:
                    if parser is None:
                        context += "- OCR 정보: 파서 모듈 없음\n"
                    else:
                        parser = ClauseDataParser()
                        ocr = parser.parse_ocr_result(request.ocr_data)
                        ocr_context = parser.create_context_from_ocr(ocr)
                        context += ocr_context + "\n"
                except Exception as e:
                    logger.warning(f"OCR 데이터 파싱 실패: {e}")
                    context += "- OCR 정보: 파싱 실패\n"
        
        # 라운드별 컨텍스트 구성
        if request.round == 1:
            # 1라운드: 초기 특약을 첫 번째로 개선
            context += f"""
## 초기 특약 (개선 대상):
제목: {request.recent_data.title}
내용: {request.recent_data.content}

## 임대인-임차인 대화 내용:
{request.recent_data.messages}

# 개선 목표: 사전조사 정보와 대화 내용을 바탕으로 초기 특약을 개선해주세요.
"""
        else:
            # 2라운드 이상: 이전 히스토리 + 현재 특약 개선
            context += f"""
## 현재 특약 (추가 개선 대상):
제목: {request.recent_data.title}
내용: {request.recent_data.content}

## 최근 대화 내용:
{request.recent_data.messages}

## 이전 특약 개선 히스토리:
"""
            for i, prev in enumerate(request.prev_data, 1):
                context += f"""
### {i}차 특약:
제목: {prev.title}
내용: {prev.content}
당시 대화: {prev.messages}
"""
            
            context += """
# 개선 목표: 사전조사 정보와 히스토리를 참고하여 현재 특약을 최근 대화 내용에 맞게 추가 개선해주세요.
"""
        
        return context
    
    @staticmethod
    def create_spring_response(improved_clause: ImprovedClause) -> Dict:
        """개선된 특약을 Spring 포맷으로 변환"""
        return {
            "round": improved_clause.round,
            "order": improved_clause.order,
            "title": improved_clause.title,
            "content": improved_clause.content,
            "assessment": {
                "owner": {
                    "level": improved_clause.owner_assessment.value,
                    "reason": improved_clause.owner_reason
                },
                "tenant": {
                    "level": improved_clause.tenant_assessment.value,
                    "reason": improved_clause.tenant_reason
                }
            }
        }


class ClauseImprovementController:
    """특약 개선 시스템 메인 컨트롤러"""
    
    def __init__(self):
        """초기화"""
        self.parser = ClauseImprovementParser()
        logger.info("ClauseImprovementParser 초기화 완료")
        
        # 모델을 필요할 때 동적으로 import
        self.improvement_model = None
    
    def _get_improvement_model(self):
        """특약 개선 모델 lazy loading"""
        if self.improvement_model is None:
            try:
                from model.improve_model import ClauseImprovementModel
                self.improvement_model = ClauseImprovementModel()
                logger.info("ClauseImprovementModel 로드 완료")
            except Exception as e:
                logger.error(f"ClauseImprovementModel 로드 실패: {e}")
                raise
        return self.improvement_model
    
    def process_clause_improvement(self, request_data: Dict) -> Dict:
        """
        특약 개선 프로세스 실행
        
        Args:
            request_data: Spring에서 받은 개선 요청 JSON
            
        Returns:
            Dict: Spring으로 반환할 개선된 특약 JSON
        """
        try:
            logger.info("특약 개선 프로세스 시작")
            
            # 1단계: 요청 데이터 파싱
            logger.info("요청 데이터 파싱 단계")
            request = self.parser.parse_improvement_request(request_data)
            
            # 2단계: 특약 개선 수행
            logger.info("특약 개선 단계 시작")
            improvement_model = self._get_improvement_model()
            improved_clause = improvement_model.improve_clause(request)
            
            if not improved_clause:
                logger.error("특약 개선 실패")
                return self._create_error_response("특약 개선에 실패했습니다.")
            
            # 3단계: Spring 응답 포맷 생성
            logger.info("응답 포맷 생성 단계")
            response = self.parser.create_spring_response(improved_clause)
            
            logger.info(f"특약 개선 완료 - 라운드: {response['round']}, 특약: {response['order']}번")
            return response
            
        except Exception as e:
            logger.error(f"특약 개선 프로세스 실패: {e}")
            return self._create_error_response(f"시스템 오류: {str(e)}")
    
    def _create_error_response(self, message: str) -> Dict:
        """에러 응답 생성"""
        return {
            "success": False,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": None
        }


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("잇집 AI - 특약 개선 시스템 테스트")
    print("=" * 60)
    
    # 테스트용 데이터
    test_request = {
        "contractChatId": 3039,
        "order": 1,
        "round": 3,
        "ownerData": {  # 임대인 사전조사 추가
            "ownerPrecheckId": 1001,
            "contractChatId": 3039,
            "identityId": 2001,
            "rentType": "JEONSE",
            "isMortgaged": True,
            "contractDuration": "2YEAR",
            "renewalIntent": "YES",
            "responseRepairingFixtures": "OWNER",
            "hasConditionLog": True,
            "hasPenalty": False,
            "hasPriorityForExtension": True,
            "hasAutoPriceAdjustment": False,
            "requireRentGuaranteeInsurance": True,
            "insuranceBurden": "PARTIAL",
            "hasNotice": "NO",
            "checkedAt": "2025-07-30T15:20:30",
            "contractFileUrl": None,
            "ownerBankName": "카카오뱅크",
            "ownerAccountNumber": "3333-12-3456789",
            "restoreCategories": [
                {"restoreCategoryId": 1, "restoreCategoryName": "벽지"},
                {"restoreCategoryId": 2, "restoreCategoryName": "가구"}
            ],
            "jeonseInfo": {
                "allowJeonseRightRegistration": True
            },
            "wolseInfo": None
        },
        "tenantData": {  # 임차인 사전조사 추가
            "contractChatId": 3039,
            "rentType": "JEONSE",
            "loanPlan": True,
            "insurancePlan": True,
            "expectedMoveInDate": "2025-08-15",
            "contractDuration": "YEAR_2",
            "renewalIntent": "YES",
            "facilityRepairNeeded": False,
            "interiorCleaningNeeded": True,
            "applianceInstallationPlan": True,
            "hasPet": True,
            "petInfo": "강아지",
            "petCount": 1,
            "indoorSmokingPlan": False,
            "earlyTerminationRisk": False,
            "requestToOwner": "반려동물과 함께 깨끗하게 거주하고 싶습니다.",
            "checkedAt": "2025-08-05T10:30:00",
            "residentCount": 2,
            "occupation": "회사원",
            "emergencyContact": "010-1234-5678",
            "relation": "배우자"
        },
        "prevData": [
            {
                "title": "반려동물 사육 관련 특약",
                "content": "임차인은 반려동물 사육을 금지한다.",
                "messages": "임대인: 반려동물은 절대 안 됩니다.\n임차인: 소형견 1마리만 키우고 싶어요."
            },
            {
                "title": "반려동물 사육 제한 특약",
                "content": "임차인은 소형견(10kg 이하) 1마리에 한해 사육할 수 있으며, 별도 보증금 50만원을 납부한다.",
                "messages": "임차인: 보증금이 너무 비싸요. 깨끗하게 사용할게요.\n임대인: 그럼 30만원으로 줄여드릴게요."
            }
        ],
        "recentData": {
            "title": "반려동물 사육 조건부 허용 특약",
            "content": "임차인은 소형견(10kg 이하) 1마리 사육 가능하며, 별도 보증금 30만원 납부 및 퇴거 시 전문업체 청소를 조건으로 한다.",
            "messages": "임차인: 전문업체 청소 비용도 많이 나올 것 같은데 좀 더 현실적으로 조정 가능할까요?\n임대인: 그럼 일반 청소 + 애완동물 털 제거 정도로 하면 어떨까요?\n임차인: 그 정도면 좋을 것 같아요. 구체적인 기준을 정해주시면 감사하겠습니다."
        }
    }
    
    # 메인 컨트롤러 실행
    controller = ClauseImprovementController()
    
    # 특약 개선 프로세스 실행
    result = controller.process_clause_improvement(test_request)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("특약 개선 결과")
    print("=" * 60)
    
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)