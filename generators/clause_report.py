"""
특약 생성을 위한 데이터 파서 및 모델 정의
Spring에서 전달받은 JSON 데이터를 파싱하고 AI 모델에서 사용할 수 있는 형태로 변환
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
import traceback
import sys
import os
from enum import Enum

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.logger_config import get_logger
logger = get_logger(__name__)


class AssessmentLevel(Enum):
    """평가 등급"""
    SAFE = "안심"
    CAUTION = "주의"
    
@dataclass
class RestoreCategory:
    """원상복구 카테고리"""
    restoreCategoryId: int
    restoreCategoryName: str


@dataclass
class JeonseInfo:
    """전세 관련 정보"""
    allowJeonseRightRegistration: bool
    
@dataclass
class WolseInfo:
    """월세 관련 정보"""
    paymentDueDay: int  #납부일 (1~31)
    lateFeeInterestRate: float  #연체 시 이자율 (% 단위, 일 기준)


@dataclass
class OwnerPrecheck:
    """임대인 사전조사 정보"""
    ownerPrecheckId: int
    contractChatId: int
    identityId: int
    rentType: str  # JEONSE, WOLSE
    isMortgaged: bool
    contractDuration: str  # 1YEAR, 2YEAR, MORE_THAN_2YEAR
    renewalIntent: str  # YES, NO, UNDECIDED
    responseRepairingFixtures: str  # OWNER, BUYER
    hasConditionLog: bool
    hasPenalty: bool
    hasPriorityForExtension: bool
    hasAutoPriceAdjustment: bool
    requireRentGuaranteeInsurance: bool
    insuranceBurden: str  # OWNER, BUYER, PARTIAL
    hasNotice: str  # YES, NO
    checkedAt: str
    contractFileUrl: Optional[str]
    ownerBankName: Optional[str]
    ownerAccountNumber: Optional[str]
    restoreCategories: List[RestoreCategory]
    jeonseInfo: Optional[JeonseInfo]
    wolseInfo: Optional[Dict] = None


@dataclass
class TenantPrecheck:
    """임차인 사전조사 정보"""
    contractChatId: int
    rentType: str  # JEONSE, WOLSE
    # 계약 기본 정보
    loanPlan: bool
    insurancePlan: bool
    expectedMoveInDate: str
    contractDuration: str  # YEAR_1, YEAR_2, YEAR_OVER_2
    renewalIntent: str  # YES, NO, UNDECIDED
    # 주거환경 요청
    facilityRepairNeeded: bool
    interiorCleaningNeeded: bool
    applianceInstallationPlan: bool
    hasPet: bool
    petInfo: Optional[str]
    petCount: Optional[int]
    # 거주자 정보
    indoorSmokingPlan: bool
    earlyTerminationRisk: bool
    requestToOwner: Optional[str]
    checkedAt: str
    residentCount: int
    occupation: str
    emergencyContact: str
    relation: str


@dataclass
class OCRResult:
    """OCR 처리 결과"""
    file_name: str
    extracted_at: str
    source: str
    special_terms: List[str]
    raw_text: str


@dataclass
class ClauseData:
    """생성된 특약 데이터"""
    id: int
    title: str
    content: str
    
@dataclass
class ClauseAssessment:
    """특약 평가 결과"""
    clause_id: int
    clause_title: str
    clause_content: str
    owner_assessment: AssessmentLevel
    owner_reason: str
    tenant_assessment: AssessmentLevel
    tenant_reason: str


class ClauseDataParser:
    """Spring JSON 데이터 파서"""
    
    @staticmethod
    def parse_owner_precheck(data: Dict) -> OwnerPrecheck:
        """임대인 사전조사 JSON 파싱"""
        restore_categories = [
            RestoreCategory(**cat) for cat in data.get('restoreCategories', [])
        ]
        
        jeonse_info = None
        if data.get('jeonseInfo'):
            jeonse_info = JeonseInfo(**data['jeonseInfo'])
        
        wolse_info = None
        if data.get('wolseInfo'):
            wolse_info = WolseInfo(**data['wolseInfo'])
        
        return OwnerPrecheck(
            ownerPrecheckId=data['ownerPrecheckId'],
            contractChatId=data['contractChatId'],
            identityId=data['identityId'],
            rentType=data['rentType'],
            isMortgaged=data['isMortgaged'],
            contractDuration=data['contractDuration'],
            renewalIntent=data['renewalIntent'],
            responseRepairingFixtures=data['responseRepairingFixtures'],
            hasConditionLog=data['hasConditionLog'],
            hasPenalty=data['hasPenalty'],
            hasPriorityForExtension=data['hasPriorityForExtension'],
            hasAutoPriceAdjustment=data['hasAutoPriceAdjustment'],
            requireRentGuaranteeInsurance=data['requireRentGuaranteeInsurance'],
            insuranceBurden=data['insuranceBurden'],
            hasNotice=data['hasNotice'],
            checkedAt=data['checkedAt'],
            contractFileUrl=data['contractFileUrl'],
            ownerBankName=data['ownerBankName'],
            ownerAccountNumber=data['ownerAccountNumber'],
            restoreCategories=restore_categories,
            jeonseInfo=jeonse_info,
            wolseInfo=wolse_info
        )
    
    @staticmethod
    def parse_tenant_precheck(data: Dict) -> TenantPrecheck:
        """임차인 사전조사 JSON 파싱"""
        return TenantPrecheck(
            contractChatId=data['contractChatId'],
            rentType=data['rentType'],
            loanPlan=data['loanPlan'],
            insurancePlan=data['insurancePlan'],
            expectedMoveInDate=data['expectedMoveInDate'],
            contractDuration=data['contractDuration'],
            renewalIntent=data['renewalIntent'],
            facilityRepairNeeded=data['facilityRepairNeeded'],
            interiorCleaningNeeded=data['interiorCleaningNeeded'],
            applianceInstallationPlan=data['applianceInstallationPlan'],
            hasPet=data['hasPet'],
            petInfo=data.get('petInfo'),
            petCount=data.get('petCount'),
            indoorSmokingPlan=data['indoorSmokingPlan'],
            earlyTerminationRisk=data['earlyTerminationRisk'],
            requestToOwner=data.get('requestToOwner'),
            checkedAt=data['checkedAt'],
            residentCount=data['residentCount'],
            occupation=data['occupation'],
            emergencyContact=data['emergencyContact'],
            relation=data['relation']
        )
    
    @staticmethod
    def parse_ocr_result(data: Dict) -> OCRResult:
        """OCR 결과 JSON 파싱"""
        return OCRResult(
            file_name=data['file_name'],
            extracted_at=data['extracted_at'],
            source=data['source'],
            special_terms=data['special_terms'],
            raw_text=data['raw_text']
        )
    
    @staticmethod
    def create_context_from_owner(owner: OwnerPrecheck) -> str:
        """임대인 정보를 컨텍스트 문자열로 변환"""
        context = f"""# 임대인 사전조사 정보:
- 임대 유형: {owner.rentType}
- 근저당 설정 여부: {'있음' if owner.isMortgaged else '없음'}
- 계약 기간: {owner.contractDuration}
- 재계약 의사: {owner.renewalIntent}
- 비품 수리 책임: {owner.responseRepairingFixtures}
- 입주 시 상태 기록: {'예' if owner.hasConditionLog else '아니오'}
- 중도 퇴거 위약금: {'있음' if owner.hasPenalty else '없음'}
- 계약 연장 우선 협의: {'있음' if owner.hasPriorityForExtension else '없음'}
- 보증보험 가입 의무: {'있음' if owner.requireRentGuaranteeInsurance else '없음'}
- 보증보험 비용 부담: {owner.insuranceBurden}
- 고지사항: {owner.hasNotice}
- 임대인 계좌: {owner.ownerBankName} {owner.ownerAccountNumber}
"""
        
        if owner.restoreCategories:
            categories = ", ".join([cat.restoreCategoryName for cat in owner.restoreCategories])
            context += f"- 원상복구 범위: {categories}\n"
        
        if owner.jeonseInfo:
            context += f"""
## 전세 관련 정보:
- 전세권 설정 허용: {'예' if owner.jeonseInfo.allowJeonseRightRegistration else '아니오'}
"""
        # 월세 관련 정보
        if owner.wolseInfo:
            context += f"""
## 월세 관련 정보:
- 월세 납부일: 매월 {owner.wolseInfo.paymentDueDay}일
- 연체 시 이자율: {owner.wolseInfo.lateFeeInterestRate}% (일 기준)
"""
        
        return context
    
    @staticmethod
    def create_context_from_tenant(tenant: TenantPrecheck) -> str:
        """임차인 정보를 컨텍스트 문자열로 변환"""
        context = f"""# 임차인 사전조사 정보:
- 임대 유형: {tenant.rentType}
- 대출 계획: {'있음' if tenant.loanPlan else '없음'}
- 보증보험 가입 계획: {'있음' if tenant.insurancePlan else '없음'}
- 입주 예정일: {tenant.expectedMoveInDate}
- 희망 계약 기간: {tenant.contractDuration}
- 재계약 의사: {tenant.renewalIntent}
- 시설 보수 필요: {'예' if tenant.facilityRepairNeeded else '아니오'}
- 도배/장판/청소 필요: {'예' if tenant.interiorCleaningNeeded else '아니오'}
- 가전 설치 계획: {'예' if tenant.applianceInstallationPlan else '아니오'}
"""
        
        if tenant.hasPet:
            context += f"- 반려동물: {tenant.petInfo} {tenant.petCount}마리\n"
        else:
            context += "- 반려동물: 없음\n"
        
        context += f"""- 실내 흡연 계획: {'있음' if tenant.indoorSmokingPlan else '없음'}
- 중도 퇴거 가능성: {'있음' if tenant.earlyTerminationRisk else '없음'}
- 거주 인원: {tenant.residentCount}명
- 직업: {tenant.occupation}
- 비상연락처: {tenant.emergencyContact} ({tenant.relation})
"""
        
        if tenant.requestToOwner:
            context += f"- 특별 요청사항: {tenant.requestToOwner}\n"
        
        return context
    
    @staticmethod
    def create_context_from_ocr(ocr: OCRResult) -> str:
        """OCR 결과를 컨텍스트 문자열로 변환"""
        context = f"""# 기존 계약서 특약 (OCR 추출):
- 파일명: {ocr.file_name}
- 추출 시간: {ocr.extracted_at}
"""
        
        if ocr.special_terms:
            context += "\n## 기존 특약 조항들:\n"
            for i, term in enumerate(ocr.special_terms, 1):
                context += f"{i}. {term}\n"
        
        else:
            context += "\n## 기존 특약: 없음\n"
                    
        return context
    
    @staticmethod
    def parse_llm_clauses_output(llm_output: str) -> List[ClauseData]:
        """LLM 출력을 ClauseData 리스트로 파싱"""
        clauses = []
        
        try:
            # "## 1번 특약\n제목: xxx\n내용: yyy" 패턴 매칭
            pattern = r'##\s*(\d+)번\s*특약\s*\n제목:\s*(.+?)\n내용:\s*(.+?)(?=##\s*\d+번\s*특약|\Z)'
            matches = re.findall(pattern, llm_output, re.DOTALL | re.MULTILINE)
            
            if matches:
                logger.info(f"OK: 정규식으로 {len(matches)}개 특약 파싱 성공")
                for i, (num, title, content) in enumerate(matches):
                    clauses.append(ClauseData(
                        id=i + 1,
                        title=title.strip(),
                        content=content.strip().replace('\n', ' ')
                    ))
            else:
                # 정규식 실패 시 수동 파싱
                logger.warning("WARNING: 정규식 파싱 실패, 수동 파싱 시도")
                clauses = ClauseDataParser._manual_parse_clauses(llm_output)
            
            logger.info(f"OK: 최종 {len(clauses)}개 특약 파싱 완료")
            return clauses
            
        except Exception as e:
            logger.error(f"ERROR: LLM 출력 파싱 실패: {e}")
            return []
        
    @staticmethod
    def _manual_parse_clauses(llm_output: str) -> List[ClauseData]:
        """수동 파싱 (정규식 실패 시 사용)"""
        clauses = []
        lines = llm_output.split('\n')
        
        current_clause = {}
        clause_id = 0
        
        for line in lines:
            line = line.strip()
            
            # "## 1번 특약" 패턴
            if re.match(r'##\s*\d+번\s*특약', line):
                # 이전 특약 저장
                if current_clause.get('제목') and current_clause.get('내용'):
                    clause_id += 1
                    
                    clauses.append(ClauseData(
                        id=clause_id,
                        title=current_clause['제목'],
                        content=current_clause['내용']
                    ))
                
                # 새 특약 시작
                current_clause = {}
                
            elif line.startswith('제목:'):
                current_clause['제목'] = line.replace('제목:', '').strip()
                
            elif line.startswith('내용:'):
                current_clause['내용'] = line.replace('내용:', '').strip()
                
            elif current_clause.get('제목') and not current_clause.get('내용') and len(line) > 20:
                # 제목이 있고 내용이 없으면서 충분히 긴 줄을 내용으로 간주
                current_clause['내용'] = line
        
        # 마지막 특약 저장
        if current_clause.get('제목') and current_clause.get('내용'):
            clause_id += 1
            
            clauses.append(ClauseData(
                id=clause_id,
                title=current_clause['제목'],
                content=current_clause['내용']
            ))
        
        return clauses
    
class ClauseReportGenerator:
    """전체 특약 생성 및 평가 시스템 - 메인 컨트롤러"""
    
    def __init__(self):
        """초기화"""
        self.parser = ClauseDataParser()
        logger.info("ClauseDataParser 초기화 완료")
        
        # 모델들을 필요할 때 동적으로 import
        self.clause_model = None
        self.assess_model = None
    
    def _get_clause_model(self):
        """특약 생성 모델 lazy loading"""
        if self.clause_model is None:
            try:
                from model.clause_model import ClauseGenerationModel
                self.clause_model = ClauseGenerationModel()
                logger.info("ClauseGenerationModel 로드 완료")
            except Exception as e:
                logger.error(f"ClauseGenerationModel 로드 실패 : {e}")
                raise
        return self.clause_model
    
    def _get_assess_model(self):
        """특약 평가 모델 lazy loading"""
        if self.assess_model is None:
            try:
                from model.assess_model import ClauseAssessmentModel
                self.assess_model = ClauseAssessmentModel()
                logger.info("ClauseAssessmentModel 로드 완료")
            except Exception as e:
                logger.error(f"ClauseAssessmentModel 로드 실패 : {e}")
                raise
        return self.assess_model

    def process_clause_generation_request(self,
                                          owner_data:Dict,
                                          tenant_data:Dict,
                                          ocr_data:Optional[Dict] = None) -> Dict:
        """
        전체 특약 생성 및 평가 프로세스 실행
        Spring -> clause_report -> clause_model -> assess_model -> clause_report -> Spring
        
        Args:
            owner_data: 임대인 사전조사 JSON  
            tenant_data: 임차인 사전조사 JSON
            ocr_data: OCR 결과 JSON (선택사항)
            
        Returns:
            Dict: Spring으로 반환할 최종 JSON
        """
        try:
            logger.info("초기 특약 생성 및 평가 프로세스")
            
            # 1단계: 특약 생성 
            logger.info("특약 생성 단계 시작")
            clause_model = self._get_clause_model()
            clauses = clause_model.generate_initial_clauses(
                owner_data = owner_data,
                tenant_data= tenant_data,
                ocr_data= ocr_data
            )
            if not clauses:
                logger.error("특약 생성 실패")
                return self._create_error_response("특약 생성 실패")

            # 2단계 : 특약 평가 
            logger.info("특약 평가 단계 시작")
            assess_model = self._get_assess_model()
            assessments = assess_model.assess_clauses(
                clauses=clauses,
                owner_data=owner_data,
                tenant_data=tenant_data
            )
            
            if not assessments:
                logger.error("특약 평가 실패")
                return self._create_error_response("특약 평가에 실패했습니다.")
            
            # 3단계 : JSON 생성
            logger.info("JSON 생성 단계 시작")
            final_json = self._create_spring_response(clauses, assessments)
            
            return final_json
            
        except Exception as e:
            logger.error(f"프로세스 실패: {e}")
            traceback.print_exc()
            return self._create_error_response(f"시스템 오류 : {str(e)}")
                
    def _create_spring_response(self, 
                                clauses: List[ClauseData],
                                assessments : List[ClauseAssessment]) -> Dict:
        """Spring으러 반환할 최종 JSON 생성"""
        
        # 평가 결과를 ID로 매핑
        assessment_map = {assessment.clause_id: assessment for assessment in assessments}
        
        # 최종 JSON 구조 생성
        response = {
            "success": True,
            "message": "특약 생성 및 평가 완료",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "total_clauses": len(clauses),
                "clauses": []
            }
        }
        
        # 각 특약과 평가 결과 결합
        for clause in clauses:
            assessment = assessment_map.get(clause.id)
            
            clause_data = {
                "order": clause.id,
                "title": clause.title,
                "content": clause.content,
                "assessment": {
                    "owner": {
                        "level": assessment.owner_assessment.value if assessment else "안심",
                        "reason": assessment.owner_reason if assessment else "평가 정보 없음"
                    },
                    "tenant": {
                        "level": assessment.tenant_assessment.value if assessment else "안심", 
                        "reason": assessment.tenant_reason if assessment else "평가 정보 없음"
                    }
                }
            }
            
            response["data"]["clauses"].append(clause_data)
        
        return response
        
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
    print("잇집 AI - 특약 생성 및 평가 시스템 테스트")
    print("=" * 60)
    
    # 테스트용 데이터
    test_owner_data = {
        "ownerPrecheckId": 1001,
        "contractChatId": 3001,
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
        "contractFileUrl": "https://your-bucket.s3.amazonaws.com/contract123.pdf",
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
    }
    
    # 월세 테스트 케이스도 추가 (필요시 사용)
    test_owner_data_wolse = {
        "ownerPrecheckId": 1002,
        "contractChatId": 3002,
        "identityId": 2002,
        "rentType": "WOLSE",
        "isMortgaged": False,
        "contractDuration": "1YEAR",
        "renewalIntent": "UNDECIDED",
        "responseRepairingFixtures": "BUYER",
        "hasConditionLog": False,
        "hasPenalty": True,
        "hasPriorityForExtension": False,
        "hasAutoPriceAdjustment": True,
        "requireRentGuaranteeInsurance": False,
        "insuranceBurden": "BUYER",
        "hasNotice": "YES",
        "checkedAt": "2025-07-30T15:20:30",
        "contractFileUrl": None,
        "ownerBankName": "국민은행",
        "ownerAccountNumber": "1234-56-789012",
        "restoreCategories": [
            {"restoreCategoryId": 3, "restoreCategoryName": "장판"}
        ],
        "jeonseInfo": None,
        "wolseInfo": {
            "paymentDueDay": 5,  # ✅ 변경된 필드명에 맞춤
            "lateFeeInterestRate": 0.05  # ✅ 변경된 필드명에 맞춤
        }
    }
    
    test_tenant_data = {
        "contractChatId": 1,
        "rentType": "JEONSE",
        "loanPlan": True,
        "insurancePlan": True,
        "expectedMoveInDate": "2025-07-22",
        "contractDuration": "YEAR_2",
        "renewalIntent": "UNDECIDED",
        "facilityRepairNeeded": False,
        "interiorCleaningNeeded": True,
        "applianceInstallationPlan": True,
        "hasPet": True,
        "petInfo": "강아지",
        "petCount": 1,
        "indoorSmokingPlan": False,
        "earlyTerminationRisk": False,
        "requestToOwner": "엘리베이터 점검일 피해서 입주 조율 가능할까요?",
        "checkedAt": "2025-07-22T10:30:00",
        "residentCount": 1,
        "occupation": "외교관",
        "emergencyContact": "010-1234-5678",
        "relation": "남편"
    }
    
    test_ocr_data = {
        "file_name": "20231006_02.pdf",
        "extracted_at": "2025-07-25T14:46:57.138249",
        "source": "text",
        "special_terms": [
            "주택을 인도받은 임차인은 _______년 ____월 ____일까지 주민등록(전입신고)과 주택임대차계약서상 확정일자를 받기로 하고, 임대인은 위 약정일자의 다음날까지 임차주택에 저당권 등 담보권을 설정할 수 없다.",
            "임대인이 위 특약에 위반하여 임차주택에 저당권 등 담보권을 설정한 경우에는 임차인은 임대차계약을 해제 또는 해지할 수 있다.",
            "주택임대차계약과 관련하여 분쟁이 있는 경우 임대인 또는 임차인은 법원에 소를 제기하기 전에 먼저 주택임대차분쟁조정위원회에 조정을 신청한다.",
        ],
        "raw_text": "전체 OCR 텍스트..."
    }
    
    # 메인 컨트롤러 실행
    generator = ClauseReportGenerator()
    
    # 전체 프로세스 실행
    result = generator.process_clause_generation_request(
        # owner_data=test_owner_data,
        owner_data=test_owner_data_wolse,
        tenant_data=test_tenant_data,
        ocr_data=test_ocr_data
    )
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("최종 결과 (Spring으로 전달될 JSON)")
    print("=" * 60)
    
    # JSON을 예쁘게 출력
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 간단한 요약 출력
    if result.get("success"):
        data = result.get("data", {})
        total_clauses = data.get("total_clauses", 0)
        statistics = data.get("statistics", {})
        
        print(f"\n요약:")
        print(f"• 총 특약 개수: {total_clauses}개")
        
        
        
        # 간단한 요약 출력
    if result and result.get("success"):
        data = result.get("data", {})
        total_clauses = data.get("total_clauses", 0)
        
        print(f"\n요약:")
        print(f"• 총 특약 개수: {total_clauses}개")
        
        # 각 특약별 상세 정보
        print(f"\n특약별 상세:")
        clauses = data.get("clauses", [])
        for clause in clauses:
            print(f"\n[{clause['order']}] {clause['title']}")
            print(f"   내용: {clause['content'][:50]}{'...' if len(clause['content']) > 50 else ''}")
            
            assessment = clause.get("assessment", {})
            owner_assess = assessment.get("owner", {})
            tenant_assess = assessment.get("tenant", {})
            
            print(f"   임대인: {owner_assess.get('level', '없음')} - {owner_assess.get('reason', '정보 없음')[:40]}{'...' if len(owner_assess.get('reason', '')) > 40 else ''}")
            print(f"   임차인: {tenant_assess.get('level', '없음')} - {tenant_assess.get('reason', '정보 없음')[:40]}{'...' if len(tenant_assess.get('reason', '')) > 40 else ''}")
    else:
        print(f"처리 실패: {result.get('message', '알 수 없는 오류') if result else 'result가 None입니다'}")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)