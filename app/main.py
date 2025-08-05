from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import os
import tempfile
import traceback
from typing import Optional, List

# UTF-8 인코딩 설정
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 프로젝트 루트 경로를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 프로젝트 모듈 import
from extractors.register_parser import extract_all_real_estate_info
from extractors.building_parser import BuildingInfoExtractor
from extractors.contract_parser import extract_special_terms
from config.logger_config import get_logger
from app.common.response import ApiResponse
from app.parsers.dto_converter import DtoConverter

# AI 모델 필수 import
from generators.risk_report import RiskReportGenerator
from generators.contract_report import ContractValidationGenerator
from generators.clause_report import ClauseReportGenerator

# 로거 설정
logger = get_logger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="잇집 AI OCR Service",
    description="부동산 문서 OCR 및 사기 위험도 분석 API",
    version="1.0.0",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "docExpansion": "list"
    },
    tags_metadata=[
        {
            "name": "시스템",
            "description": "서비스 상태 확인 및 시스템 정보"
        },
        {
            "name": "OCR 분석",
            "description": "부동산 문서 OCR 분석 기능 (등기부등본, 건축물대장, 계약서)"
        },
        {
            "name": "위험도 분석", 
            "description": "부동산 사기 위험도 종합 분석"
        }
    ]
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 건축물대장 파서 인스턴스 생성
building_extractor = BuildingInfoExtractor()

# 위험도 분석기 인스턴스 생성 (필수)
risk_report_generator = None
contract_validation_generator = None
clause_report_generator = None
try:
    # 환경 변수 확인
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is not set in environment variables")
    if not os.getenv("JUSO_API_KEY"):
        logger.warning("JUSO_API_KEY is not set - address verification will use fallback method")
    
    risk_report_generator = RiskReportGenerator()
    logger.info("Risk analysis model loaded successfully")
    
    contract_validation_generator = ContractValidationGenerator()
    logger.info("Contract validation model loaded successfully")
    
    clause_report_generator = ClauseReportGenerator()
    logger.info("Clause report generator loaded successfully")
except Exception as e:
    logger.error(f"Failed to load AI models: {e}")
    logger.error("Please check your environment variables and dependencies")
    # AI 모델은 필수이므로 서버 시작을 중단
    raise RuntimeError(f"Cannot start server without AI models: {e}")


# Request Models
class MortgageeInfo(BaseModel):
    """근저당권 정보"""
    priority_number: int = Field(..., alias="priorityNumber", description="순위번호", example=1)
    max_claim_amount: Optional[int] = Field(None, alias="maxClaimAmount", description="채권최고액 (원)", example=300000000)
    debtor: str = Field(..., description="채무자", example="홍길동")
    mortgagee: Optional[str] = Field(None, description="근저당권자", example="국민은행")

    model_config = ConfigDict(populate_by_name=True)


class RegistryDocumentDto(BaseModel):
    """등기부등본 정보 DTO
    
    등기부등본에서 추출한 정보를 담는 데이터 전송 객체입니다.
    """
    region_address: str = Field(..., alias="regionAddress", description="소재지번 주소", example="서울특별시 강남구 대치동 123-45")
    road_address: str = Field("", alias="roadAddress", description="도로명주소", example="서울특별시 강남구 테헤란로 123")
    owner_name: str = Field(..., alias="ownerName", description="소유자명", example="홍길동")
    owner_birth_date: Optional[str] = Field(None, alias="ownerBirthDate", description="소유자 생년월일", example="1970-01-01")
    debtor: Optional[str] = Field(None, description="채무자", example="홍길동")
    mortgagee_list: Optional[List[MortgageeInfo]] = Field(None, alias="mortgageeList", description="근저당권 목록 (순위번호, 채권최고액, 채무자, 근저당권자)")
    has_seizure: bool = Field(False, alias="hasSeizure", description="가압류 여부", example=False)
    has_auction: bool = Field(False, alias="hasAuction", description="경매 여부", example=False)
    has_litigation: bool = Field(False, alias="hasLitigation", description="소송 여부", example=False)
    has_attachment: bool = Field(False, alias="hasAttachment", description="압류 여부", example=False)

    model_config = ConfigDict(populate_by_name=True)


class BuildingDocumentDto(BaseModel):
    """건축물대장 정보 DTO
    
    건축물대장에서 추출한 정보를 담는 데이터 전송 객체입니다.
    """
    site_location: str = Field(..., alias="siteLocation", description="대지위치", example="서울특별시 강남구 대치동 123-45")
    road_address: str = Field("", alias="roadAddress", description="도로명주소", example="서울특별시 강남구 테헤란로 123")
    total_floor_area: float = Field(..., alias="totalFloorArea", description="연면적 (㎡)", example=84.5)
    purpose: str = Field("", description="건물 용도", example="아파트")
    floor_number: int = Field(0, alias="floorNumber", description="층수", example=15)
    approval_date: Optional[str] = Field(None, alias="approvalDate", description="사용승인일 (YYYY.MM.DD)", example="2010-05-15")
    is_violation_building: bool = Field(False, alias="isViolationBuilding", description="위반건축물 여부", example=False)

    model_config = ConfigDict(populate_by_name=True)


class RiskAnalysisRequest(BaseModel):
    """위험도 분석 요청 모델
    
    부동산 사기 위험도 분석을 위한 요청 데이터입니다.
    사용자 정보, 매물 정보, 그리고 등기부등본/건축물대장 정보를 포함합니다.
    """
    user_id: int = Field(..., alias="userId", description="사용자 ID", example=123)
    user_type: str = Field(..., alias="userType", description="사용자 타입 (landlord: 임대인, tenant: 임차인)", example="tenant")
    home_id: int = Field(..., alias="homeId", description="매물 ID", example=456)
    address: str = Field(..., description="매물 주소", example="서울특별시 강남구 테헤란로 123")
    property_price: Optional[int] = Field(None, alias="propertyPrice", description="매물 가격 (전세: 전세금, 월세: 보증금)", example=500000000)
    monthly_rent: Optional[int] = Field(None, alias="monthlyRent", description="월세 금액 (원) - 월세인 경우에만", example=1000000)
    lease_type: Optional[str] = Field(None, alias="leaseType", description="임대 유형 (JEONSE: 전세, WOLSE: 월세)", example="JEONSE")
    registry_document: RegistryDocumentDto = Field(..., alias="registryDocument", description="등기부등본 정보")
    building_document: BuildingDocumentDto = Field(..., alias="buildingDocument", description="건축물대장 정보")
    registered_user_name: str = Field(..., alias="registeredUserName", description="매물 등록자 이름", example="김철수")
    residence_type: str = Field(..., alias="residenceType", description="주거 타입 (APARTMENT, OFFICETEL 등)", example="APARTMENT")

    model_config = ConfigDict(populate_by_name=True)


class ContractValidationRequest(BaseModel):
    """계약서 검증 요청 모델
    
    Spring backend에서 전달하는 계약서 정보를 담는 데이터 전송 객체입니다.
    """
    contract_id: int = Field(..., alias="contractId", description="계약서 ID", example=1)
    home_id: int = Field(..., alias="homeId", description="매물 ID", example=100)
    owner_id: int = Field(..., alias="ownerId", description="임대인 ID", example=10)
    buyer_id: int = Field(..., alias="buyerId", description="임차인 ID", example=20)
    contract_date: str = Field(..., alias="contractDate", description="계약 날짜 (YYYY-MM-DD)", example="2024-01-01")
    contract_expire_date: str = Field(..., alias="contractExpireDate", description="계약 만료 날짜 (YYYY-MM-DD)", example="2025-12-31")
    deposit_price: Optional[int] = Field(None, alias="depositPrice", description="보증금 (원)", example=200000000)
    monthly_rent: Optional[int] = Field(None, alias="monthlyRent", description="월세 (원)", example=1000000)
    maintenance_fee: Optional[int] = Field(None, alias="maintenanceFee", description="관리비 (원)", example=150000)
    special_clauses: List[str] = Field(..., alias="specialClauses", description="특약사항 목록", 
                                       example=[
                                           "임차인은 계약 해지 시 원상복구 비용을 전액 부담한다.",
                                           "애완동물 사육을 허가하되, 추가 보증금 50만원을 납부한다.",
                                           "임대인은 언제든지 3일 전 통보로 계약을 해지할 수 있다.",
                                           "임차인은 전대 및 양도를 할 수 없다.",
                                           "월세를 3일 이상 연체 시 연체료는 일 1%로 한다."
                                       ])

    model_config = ConfigDict(populate_by_name=True)


class RestoreCategoryInfo(BaseModel):
    """원상복구 카테고리 정보"""
    restore_category_id: int = Field(..., alias="restoreCategoryId", description="원상복구 카테고리 ID", example=1)
    restore_category_name: str = Field(..., alias="restoreCategoryName", description="원상복구 카테고리명", example="벽지")
    
    model_config = ConfigDict(populate_by_name=True)


class JeonseInfoDto(BaseModel):
    """전세 관련 정보"""
    allow_jeonse_right_registration: bool = Field(..., alias="allowJeonseRightRegistration", 
                                                  description="전세권 설정 허용 여부", example=True)
    
    model_config = ConfigDict(populate_by_name=True)


class WolseInfoDto(BaseModel):
    """월세 관련 정보"""
    payment_due_day: int = Field(..., alias="paymentDueDay", 
                                description="월세 납부일 (1~31)", example=5)
    late_fee_interest_rate: float = Field(..., alias="lateFeeInterestRate", 
                                         description="연체 시 이자율 (% 단위, 일 기준)", example=0.05)
    
    model_config = ConfigDict(populate_by_name=True)


class OwnerPrecheckDto(BaseModel):
    """임대인 사전조사 정보"""
    owner_precheck_id: int = Field(..., alias="ownerPrecheckId", description="임대인 사전조사 ID", example=1001)
    contract_chat_id: int = Field(..., alias="contractChatId", description="계약 채팅방 ID", example=3001)
    identity_id: int = Field(..., alias="identityId", description="신원 ID", example=2001)
    rent_type: str = Field(..., alias="rentType", description="임대 유형 (JEONSE, WOLSE)", example="JEONSE")
    is_mortgaged: bool = Field(..., alias="isMortgaged", description="근저당 설정 여부", example=True)
    contract_duration: str = Field(..., alias="contractDuration", 
                                  description="계약 기간 (1YEAR, 2YEAR, MORE_THAN_2YEAR)", example="2YEAR")
    renewal_intent: str = Field(..., alias="renewalIntent", 
                               description="재계약 의사 (YES, NO, UNDECIDED)", example="YES")
    response_repairing_fixtures: str = Field(..., alias="responseRepairingFixtures", 
                                           description="비품 수리 책임 (OWNER, BUYER)", example="OWNER")
    has_condition_log: bool = Field(..., alias="hasConditionLog", description="입주 시 상태 기록 여부", example=True)
    has_penalty: bool = Field(..., alias="hasPenalty", description="중도 퇴거 위약금 여부", example=False)
    has_priority_for_extension: bool = Field(..., alias="hasPriorityForExtension", 
                                           description="계약 연장 우선 협의 여부", example=True)
    has_auto_price_adjustment: bool = Field(..., alias="hasAutoPriceAdjustment", 
                                          description="자동 가격 조정 여부", example=False)
    require_rent_guarantee_insurance: bool = Field(..., alias="requireRentGuaranteeInsurance", 
                                                  description="임대차 보증보험 가입 의무", example=True)
    insurance_burden: str = Field(..., alias="insuranceBurden", 
                                 description="보험 비용 부담 (OWNER, BUYER, PARTIAL)", example="PARTIAL")
    has_notice: str = Field(..., alias="hasNotice", description="고지사항 유무 (YES, NO)", example="NO")
    checked_at: str = Field(..., alias="checkedAt", description="조사 일시", example="2025-07-30T15:20:30")
    contract_file_url: Optional[str] = Field(None, alias="contractFileUrl", 
                                           description="계약서 파일 URL", 
                                           example="https://your-bucket.s3.amazonaws.com/contract123.pdf")
    owner_bank_name: Optional[str] = Field(None, alias="ownerBankName", description="임대인 은행명", example="카카오뱅크")
    owner_account_number: Optional[str] = Field(None, alias="ownerAccountNumber", 
                                              description="임대인 계좌번호", example="3333-12-3456789")
    restore_categories: List[RestoreCategoryInfo] = Field(..., alias="restoreCategories", 
                                                         description="원상복구 카테고리 목록")
    jeonse_info: Optional[JeonseInfoDto] = Field(None, alias="jeonseInfo", description="전세 관련 정보")
    wolse_info: Optional[WolseInfoDto] = Field(None, alias="wolseInfo", description="월세 관련 정보")
    
    model_config = ConfigDict(populate_by_name=True)


class TenantPrecheckDto(BaseModel):
    """임차인 사전조사 정보"""
    contract_chat_id: int = Field(..., alias="contractChatId", description="계약 채팅방 ID", example=1)
    rent_type: str = Field(..., alias="rentType", description="임대 유형 (JEONSE, WOLSE)", example="JEONSE")
    loan_plan: bool = Field(..., alias="loanPlan", description="대출 계획 여부", example=True)
    insurance_plan: bool = Field(..., alias="insurancePlan", description="보증보험 가입 계획", example=True)
    expected_move_in_date: str = Field(..., alias="expectedMoveInDate", 
                                      description="입주 예정일", example="2025-07-22")
    contract_duration: str = Field(..., alias="contractDuration", 
                                  description="계약 기간 (YEAR_1, YEAR_2, YEAR_OVER_2)", example="YEAR_2")
    renewal_intent: str = Field(..., alias="renewalIntent", 
                               description="재계약 의사 (YES, NO, UNDECIDED)", example="UNDECIDED")
    facility_repair_needed: bool = Field(..., alias="facilityRepairNeeded", 
                                       description="시설 보수 필요 여부", example=False)
    interior_cleaning_needed: bool = Field(..., alias="interiorCleaningNeeded", 
                                         description="도배/장판/청소 필요 여부", example=True)
    appliance_installation_plan: bool = Field(..., alias="applianceInstallationPlan", 
                                            description="가전 설치 계획", example=True)
    has_pet: bool = Field(..., alias="hasPet", description="반려동물 유무", example=True)
    pet_info: Optional[str] = Field(None, alias="petInfo", description="반려동물 정보", example="강아지")
    pet_count: Optional[int] = Field(None, alias="petCount", description="반려동물 수", example=1)
    indoor_smoking_plan: bool = Field(..., alias="indoorSmokingPlan", description="실내 흡연 계획", example=False)
    early_termination_risk: bool = Field(..., alias="earlyTerminationRisk", 
                                       description="중도 퇴거 가능성", example=False)
    request_to_owner: Optional[str] = Field(None, alias="requestToOwner", 
                                          description="임대인에게 특별 요청사항", 
                                          example="엘리베이터 점검일 피해서 입주 조율 가능할까요?")
    checked_at: str = Field(..., alias="checkedAt", description="조사 일시", example="2025-07-22T10:30:00")
    resident_count: int = Field(..., alias="residentCount", description="거주 인원", example=1)
    occupation: str = Field(..., description="직업", example="외교관")
    emergency_contact: str = Field(..., alias="emergencyContact", description="비상연락처", example="010-1234-5678")
    relation: str = Field(..., description="관계", example="남편")
    
    model_config = ConfigDict(populate_by_name=True)


class OcrResultDto(BaseModel):
    """OCR 처리 결과"""
    file_name: str = Field(..., alias="fileName", description="파일명", example="20231006_02.pdf")
    extracted_at: str = Field(..., alias="extractedAt", description="추출 시간", example="2025-07-25T14:46:57.138249")
    source: str = Field(..., description="추출 방식", example="text")
    special_terms: List[str] = Field(..., alias="specialTerms", description="특약사항 목록")
    raw_text: str = Field(..., alias="rawText", description="원본 텍스트", example="전체 OCR 텍스트...")
    
    model_config = ConfigDict(populate_by_name=True)


class ClauseRecommendationRequest(BaseModel):
    """특약 추천 요청 모델"""
    owner_data: OwnerPrecheckDto = Field(..., alias="ownerData", description="임대인 사전조사 정보")
    tenant_data: TenantPrecheckDto = Field(..., alias="tenantData", description="임차인 사전조사 정보")
    ocr_data: Optional[OcrResultDto] = Field(None, alias="ocrData", description="OCR 결과 (선택사항)")
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "ownerData": {
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
                        {
                            "restoreCategoryId": 3,
                            "restoreCategoryName": "장판"
                        }
                    ],
                    "jeonseInfo": None,
                    "wolseInfo": {
                        "paymentDueDay": 5,
                        "lateFeeInterestRate": 0.05
                    }
                },
                "tenantData": {
                    "contractChatId": 3002,
                    "rentType": "WOLSE",
                    "loanPlan": False,
                    "insurancePlan": False,
                    "expectedMoveInDate": "2025-08-01",
                    "contractDuration": "YEAR_1",
                    "renewalIntent": "UNDECIDED",
                    "facilityRepairNeeded": True,
                    "interiorCleaningNeeded": False,
                    "applianceInstallationPlan": False,
                    "hasPet": True,
                    "petInfo": "고양이",
                    "petCount": 2,
                    "indoorSmokingPlan": True,
                    "earlyTerminationRisk": True,
                    "requestToOwner": "창문 방충망 교체가 필요하고, 에어컨 점검도 부탁드립니다.",
                    "checkedAt": "2025-07-25T10:30:00",
                    "residentCount": 2,
                    "occupation": "IT 개발자",
                    "emergencyContact": "010-9876-5432",
                    "relation": "배우자"
                },
                "ocrData": {
                    "fileName": "20250725_contract.pdf",
                    "extractedAt": "2025-07-25T14:46:57.138249",
                    "source": "text",
                    "specialTerms": [
                        "임차인은 계약 종료 시 임차목적물을 원상회복하여 임대인에게 반환하여야 한다.",
                        "월 임대료는 매월 5일까지 임대인이 지정한 계좌로 입금하여야 한다."
                    ],
                    "rawText": "부동산 임대차 계약서..."
                }
            }
        }
    )


@app.get("/", 
         summary="API 정보",
         description="API 서비스 정보 및 사용 가능한 엔드포인트 목록을 제공합니다.",
         tags=["시스템"])
async def root():
    """루트 엔드포인트"""
    data = {
        "message": "잇집 AI OCR Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "docs": "/docs",
            "parse_register": "/api/parse/register",
            "parse_building": "/api/parse/building",
            "parse_contract": "/api/parse/contract",
            "analyze_risk": "/api/analyze/risk",
            "validate_contract": "/api/contract/validate"
        }
    }
    return ApiResponse.success(data=data)


@app.get("/api/health", 
         summary="서비스 상태 확인",
         description="서비스가 정상적으로 동작 중인지 확인합니다.",
         tags=["시스템"])
async def health_check():
    """헬스 체크 엔드포인트"""
    data = {
        "status": "healthy",
        "service": "itzip-ai-ocr"
    }
    return ApiResponse.success(data=data, message="Service is healthy")


@app.post("/api/parse/register",
          summary="등기부등본 OCR 분석",
          description="등기부등본 PDF 파일을 업로드하면 OCR을 통해 주요 정보를 추출합니다.\n\n추출 정보:\n- 소재지번 및 도로명주소\n- 소유자 정보\n- 권리관계 (갑구/을구)\n- 법적 상태 (가압류, 경매, 소송 등)",
          tags=["OCR 분석"],
          response_model=ApiResponse)
async def parse_register(file: UploadFile = File(..., description="등기부등본 PDF 파일")):
    """등기부등본 파싱 API"""
    if not file.filename.endswith('.pdf'):
        return ApiResponse.error(
            message="PDF 파일만 업로드 가능합니다.",
            code="INVALID_FILE_TYPE",
            field="file"
        )
    
    temp_file_path = None
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        
        logger.info(f"등기부등본 파싱 시작: {file.filename}")
        
        try:
            # 등기부등본 정보 추출
            result = extract_all_real_estate_info(temp_file_path)
            
            logger.info(f"등기부등본 파싱 완료: {file.filename}")
            
            # Spring DTO 형식으로 변환
            dto_result = DtoConverter.convert_register_to_dto(result)
            
            data = {
                "filename": file.filename,
                "document_type": "register",
                "parsed_data": dto_result
            }
            
            return ApiResponse.success(
                data=data,
                message="등기부등본 파싱이 완료되었습니다."
            )
        except ValueError as ve:
            # 등기부등본이 아닌 경우
            logger.warning(f"등기부등본 유효성 검증 실패: {str(ve)}")
            return ApiResponse.error(
                message=str(ve),
                code="INVALID_DOCUMENT_TYPE",
                field="file"
            )
        
    except Exception as e:
        logger.error(f"등기부등본 파싱 오류: {str(e)}")
        return ApiResponse.error(
            message=f"파싱 중 오류 발생: {str(e)}",
            code="PARSING_ERROR"
        )
    
    finally:
        # 임시 파일 삭제
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.post("/api/parse/building",
          summary="건축물대장 OCR 분석",
          description="건축물대장 PDF 파일을 업로드하면 OCR을 통해 주요 정보를 추출합니다.\n\n추출 정보:\n- 대지위치 및 도로명주소\n- 연면적 및 층수\n- 건물 용도\n- 사용승인일\n- 위반건축물 여부",
          tags=["OCR 분석"],
          response_model=ApiResponse)
async def parse_building(file: UploadFile = File(..., description="건축물대장 PDF 파일")):
    """건축물대장 파싱 API"""
    if not file.filename.endswith('.pdf'):
        return ApiResponse.error(
            message="PDF 파일만 업로드 가능합니다.",
            code="INVALID_FILE_TYPE",
            field="file"
        )
    
    temp_file_path = None
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        
        logger.info(f"건축물대장 파싱 시작: {file.filename}")
        
        try:
            # 건축물대장 정보 추출
            result = building_extractor.extract_building_info_from_crop(temp_file_path)
            
            logger.info(f"건축물대장 파싱 완료: {file.filename}")
            
            # Spring DTO 형식으로 변환
            dto_result = DtoConverter.convert_building_to_dto(result)
            
            data = {
                "filename": file.filename,
                "document_type": "building",
                "parsed_data": dto_result
            }
            
            return ApiResponse.success(
                data=data,
                message="건축물대장 파싱이 완료되었습니다."
            )
        except ValueError as ve:
            # 건축물대장이 아닌 경우
            logger.warning(f"건축물대장 유효성 검증 실패: {str(ve)}")
            return ApiResponse.error(
                message=str(ve),
                code="INVALID_DOCUMENT_TYPE",
                field="file"
            )
        
    except Exception as e:
        logger.error(f"건축물대장 파싱 오류: {str(e)}")
        return ApiResponse.error(
            message=f"파싱 중 오류 발생: {str(e)}",
            code="PARSING_ERROR"
        )
    
    finally:
        # 임시 파일 삭제
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.post("/api/parse/contract",
          summary="임대차계약서 특약사항 추출",
          description="임대차계약서 PDF 파일에서 특약사항을 추출합니다.\n\n추출 정보:\n- 특약사항 목록\n- 추출 방식 (텍스트/이미지 기반)",
          tags=["OCR 분석"],
          response_model=ApiResponse)
async def parse_contract(file: UploadFile = File(..., description="임대차계약서 PDF 파일")):
    """계약서 파싱 API"""
    if not file.filename.endswith('.pdf'):
        return ApiResponse.error(
            message="PDF 파일만 업로드 가능합니다.",
            code="INVALID_FILE_TYPE",
            field="file"
        )
    
    temp_file_path = None
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        
        logger.info(f"계약서 파싱 시작: {file.filename}")
        
        # 계약서 정보 추출
        result = extract_special_terms(temp_file_path)
        
        logger.info(f"계약서 파싱 완료: {file.filename}")
        
        data = {
            "filename": file.filename,
            "document_type": "contract",
            "parsed_data": result
        }
        
        return ApiResponse.success(
            data=data,
            message="계약서 파싱이 완료되었습니다."
        )
        
    except Exception as e:
        logger.error(f"계약서 파싱 오류: {str(e)}")
        return ApiResponse.error(
            message=f"파싱 중 오류 발생: {str(e)}",
            code="PARSING_ERROR"
        )
    
    finally:
        # 임시 파일 삭제
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.post("/api/analyze/risk",
          summary="부동산 사기 위험도 분석",
          description="""등기부등본과 건축물대장 정보를 기반으로 부동산 사기 위험도를 분석합니다.

분석 항목:
- 소유자 일치 여부 확인
- 권리관계 분석 (근저당, 가압류, 경매 등)
- 건물 안전성 검토 (위반건축물, 노후도 등)
- 가격 적정성 평가

위험도 등급:
- **SAFE** (안전): 특별한 위험 요소가 발견되지 않음
- **WARN** (주의): 주의가 필요한 사항이 있음
- **DANGER** (위험): 즉시 조치가 필요한 위험 요소 발견

분석 결과는 4개 카테고리로 구분하여 상세 정보를 제공합니다:
1. 소유자 정보 분석
2. 권리관계 분석
3. 건물 상태 분석
4. 종합 위험도 평가""",
          tags=["위험도 분석"],
          response_model=ApiResponse)
async def analyze_risk(request: RiskAnalysisRequest):
    """위험도 분석 API"""
    try:
        logger.info(f"위험도 분석 시작: user_id={request.user_id}, home_id={request.home_id}")
        
        # DTO 변환
        registry_dto = request.registry_document.model_dump(by_alias=True)
        building_dto = request.building_document.model_dump(by_alias=True)
        
        # 위험도 분석 수행 (전역 인스턴스 사용)
        result = risk_report_generator.generate_spring_risk_report(
            user_id=request.user_id,
            user_type=request.user_type,
            home_id=request.home_id,
            address=request.address,
            property_price=request.property_price,
            lease_type=request.lease_type,
            spring_registry_dto=registry_dto,
            spring_building_dto=building_dto,
            registered_user_name=request.registered_user_name,
            residence_type=request.residence_type,
            monthly_rent=request.monthly_rent
        )
        
        logger.info(f"위험도 분석 완료: risk_type={result.get('riskType')}")
        
        return ApiResponse.success(
            data=result,
            message="위험도 분석이 완료되었습니다."
        )
        
    except Exception as e:
        logger.error(f"위험도 분석 오류: {str(e)}")
        return ApiResponse.error(
            message=f"위험도 분석 중 오류 발생: {str(e)}",
            code="RISK_ANALYSIS_ERROR"
        )


@app.post("/api/contract/validate",
          summary="계약서 법령 적법성 검증",
          description="""계약서 특약사항의 법령 적법성을 검증합니다.

주요 기능:
- 주택임대차보호법 등 관련 법령과 대조하여 위반사항 검토
- 문제가 있는 조항에 대한 상세 설명 제공
- 법적 근거 및 개선 방안 제시

검증 결과:
- **LEGAL** (적법): 법령 위반사항이 없음
- **CAUTION** (주의): 주의가 필요한 조항이 있음
- **VIOLATION** (위반): 명백한 법령 위반이 있음

각 위반사항에 대해:
- 위반 유형 및 관련 법령
- 문제가 되는 내용과 설명
- 법적 근거 (조항)
- 개선 방안 예시""",
          tags=["위험도 분석"],
          response_model=ApiResponse,
          responses={
              200: {
                  "description": "계약서 검증 성공",
                  "content": {
                      "application/json": {
                          "example": {
                              "success": True,
                              "message": "계약서 법령 검증이 완료되었습니다.",
                              "data": {
                                  "success": True,
                                  "contract_id": 1,
                                  "validation_status": "VIOLATION",
                                  "total_violations": 2,
                                  "violation_summary": {
                                      "illegal_count": 1,
                                      "caution_count": 1
                                  },
                                  "violations": [
                                      {
                                          "violation_type": "위반",
                                          "law_name": "주택임대차보호법",
                                          "violation_content": "임대인은 언제든지 3일 전 통보로 계약을 해지할 수 있다",
                                          "explanation": "임대인의 일방적 계약 해지는 법적으로 보장된 임차인의 권리를 침해합니다. 주택임대차보호법은 임차인의 주거 안정을 위해 계약 해지 사유를 엄격히 제한하고 있습니다.",
                                          "legal_basis": "제6조의3 (계약갱신 요구 등)",
                                          "improvement_example": "계약 해지는 다음의 경우에 한하여 1개월 전 서면 통지로 가능합니다: 1) 임차인이 2기 이상 차임을 연체한 경우, 2) 임차인이 임대인의 동의 없이 전대한 경우, 3) 기타 법정 해지 사유에 해당하는 경우",
                                          "original_clause": "임대인은 언제든지 3일 전 통보로 계약을 해지할 수 있다."
                                      },
                                      {
                                          "violation_type": "주의",
                                          "law_name": "민법",
                                          "violation_content": "월세를 3일 이상 연체 시 연체료는 일 1%로 한다",
                                          "explanation": "일 1%의 연체료는 연 365%에 해당하는 과도한 이율입니다. 법정 최고이자율을 초과하는 약정은 무효가 될 수 있습니다.",
                                          "legal_basis": "제2조 (이자의 제한)",
                                          "improvement_example": "월세 연체 시 연체료는 연 12% 이내에서 일할 계산하여 부과합니다.",
                                          "original_clause": "월세를 3일 이상 연체 시 연체료는 일 1%로 한다."
                                      }
                                  ],
                                  "validated_at": "2024-01-01T12:00:00",
                                  "recommendation": "법령 위반 조항 1건이 발견되어 계약서 수정이 필요합니다. 전문가 상담을 권장드립니다."
                              },
                              "error": None,
                              "timestamp": "2024-01-01T12:00:00"
                          }
                      }
                  }
              },
              400: {
                  "description": "잘못된 요청",
                  "content": {
                      "application/json": {
                          "example": {
                              "success": False,
                              "message": "요청 데이터가 올바르지 않습니다.",
                              "data": None,
                              "error": {
                                  "code": "INVALID_REQUEST",
                                  "field": "contractDate",
                                  "rejectedValue": "2024-13-01",
                                  "reason": "올바른 날짜 형식이 아닙니다."
                              },
                              "timestamp": "2024-01-01T12:00:00"
                          }
                      }
                  }
              }
          })
async def validate_contract(request: ContractValidationRequest):
    """계약서 법령 검증 API"""
    try:
        logger.info(f"계약서 검증 시작: contract_id={request.contract_id}")
        
        # Request DTO를 generator가 기대하는 형식으로 변환
        contract_data = request.model_dump(by_alias=True)
        
        # 계약서 검증 수행
        result = contract_validation_generator.validate_contract_for_spring(contract_data)
        
        logger.info(f"계약서 검증 완료: status={result.get('validation_status')}, violations={result.get('total_violations')}")
        
        return ApiResponse.success(
            data=result,
            message="계약서 법령 검증이 완료되었습니다."
        )
        
    except Exception as e:
        logger.error(f"계약서 검증 오류: {str(e)}")
        return ApiResponse.error(
            message=f"계약서 검증 중 오류 발생: {str(e)}",
            code="CONTRACT_VALIDATION_ERROR"
        )


@app.post("/api/clause/recommend",
          summary="신규 특약 추천",
          description="""임대인과 임차인의 사전조사 정보를 기반으로 맞춤형 특약 6개를 추천합니다.

주요 기능:
- 임대인 사전조사 정보 분석 (임대 조건, 보증보험, 원상복구 등)
- 임차인 사전조사 정보 분석 (반려동물, 흡연, 거주 환경 등)
- 기존 OCR 특약 분석 (선택사항)
- AI 기반 맞춤형 특약 6개 생성
- 각 특약에 대한 양측 이익 평가 (안심/주의)

분석 결과:
- 생성된 특약 목록 (제목, 내용)
- 임대인 관점 평가 (안심/주의 + 사유)
- 임차인 관점 평가 (안심/주의 + 사유)

특약 생성 시 고려사항:
- 전세/월세 유형별 차별화
- 법령 근거 기반 조항 생성
- 양측 이익 균형 반영
- 구체적이고 실행 가능한 내용""",
          tags=["특약 추천"],
          response_model=ApiResponse,
          responses={
              200: {
                  "description": "특약 추천 성공",
                  "content": {
                      "application/json": {
                          "examples": {
                              "월세 계약 예시": {
                                  "value": {
                                      "success": True,
                                      "message": "특약 생성 및 평가 완료",
                                      "data": {
                                          "timestamp": "2025-07-30T15:45:30.123456",
                                          "total_clauses": 6,
                                          "clauses": [
                                              {
                                                  "order": 1,
                                                  "title": "반려동물 및 흡연 관련 특약",
                                                  "content": "임차인은 고양이 2마리를 사육할 수 있으며, 실내 흡연은 베란다에서만 허용된다. 반려동물로 인한 벽지, 바닥재 손상 및 흡연으로 인한 도배 변색 시 임차인이 원상복구 비용을 부담한다.",
                                                  "assessment": {
                                                      "owner": {
                                                          "level": "주의",
                                                          "reason": "반려동물과 흡연으로 인한 손상 가능성이 높아 원상복구 비용이 증가할 수 있습니다. 구체적인 손상 범위와 비용 산정 기준을 명확히 하는 것이 필요합니다."
                                                      },
                                                      "tenant": {
                                                          "level": "안심",
                                                          "reason": "반려동물 사육과 제한적 흡연이 허용되어 생활의 자유가 보장되며, 책임 범위가 명확히 규정되어 있습니다."
                                                      }
                                                  }
                                              },
                                              {
                                                  "order": 2,
                                                  "title": "월세 납부 및 연체료 특약",
                                                  "content": "월세는 매월 5일까지 임대인 지정 계좌로 납부하며, 연체 시 일 0.05%의 연체료가 부과된다. 단, 3일 이내 납부 시 연체료는 면제된다.",
                                                  "assessment": {
                                                      "owner": {
                                                          "level": "안심",
                                                          "reason": "명확한 납부일과 합리적인 연체료 규정으로 안정적인 임대수익을 확보할 수 있습니다."
                                                      },
                                                      "tenant": {
                                                          "level": "주의",
                                                          "reason": "연체료 부담이 있으나, 일 0.05%는 연 18.25%로 법정 한도 내의 수준이며 3일의 유예기간이 있어 급작스러운 부담은 완화됩니다."
                                                      }
                                                  }
                                              },
                                              {
                                                  "order": 3,
                                                  "title": "시설 수리 책임 특약",
                                                  "content": "보일러, 에어컨 등 기존 설비의 노후로 인한 고장은 임대인이 수리하며, 임차인의 과실로 인한 고장은 임차인이 부담한다. 입주 전 보일러 점검은 임대인이 실시하고, 방충망 교체는 임차인이 부담한다.",
                                                  "assessment": {
                                                      "owner": {
                                                          "level": "안심",
                                                          "reason": "노후 설비에 대한 책임만 부담하고 임차인 과실은 면책되어 합리적인 책임 분담이 이루어집니다."
                                                      },
                                                      "tenant": {
                                                          "level": "안심",
                                                          "reason": "기본 설비의 노후 고장은 임대인이 책임지므로 예상치 못한 수리비 부담이 줄어듭니다."
                                                      }
                                                  }
                                              },
                                              {
                                                  "order": 4,
                                                  "title": "중도 해지 특약",
                                                  "content": "임차인의 불가피한 사정(해외 발령, 질병 등)으로 중도 해지 시 1개월 전 통보하면 위약금 없이 계약 해지가 가능하다. 단, 임차인은 새로운 임차인을 구하는데 협조해야 한다.",
                                                  "assessment": {
                                                      "owner": {
                                                          "level": "주의",
                                                          "reason": "중도 해지 가능성으로 인해 공실 위험이 있으나, 새 임차인 구하기 협조 조항으로 리스크가 일부 완화됩니다."
                                                      },
                                                      "tenant": {
                                                          "level": "안심",
                                                          "reason": "예상치 못한 상황 발생 시 위약금 부담 없이 계약 해지가 가능하여 유연한 주거 계획이 가능합니다."
                                                      }
                                                  }
                                              },
                                              {
                                                  "order": 5,
                                                  "title": "원상복구 범위 특약",
                                                  "content": "계약 종료 시 바닥재, 싱크대, 도배는 임차인이 원상복구한다. 단, 통상적인 사용으로 인한 자연 마모는 제외하며, 입주 시 시설물 상태를 사진으로 기록하여 양 당사자가 보관한다.",
                                                  "assessment": {
                                                      "owner": {
                                                          "level": "안심",
                                                          "reason": "원상복구 범위가 명확하고 사진 증빙으로 분쟁 소지가 줄어들어 재산 보호에 유리합니다."
                                                      },
                                                      "tenant": {
                                                          "level": "안심",
                                                          "reason": "자연 마모는 제외되고 입주 시 상태 기록으로 부당한 원상복구 요구를 방지할 수 있습니다."
                                                      }
                                                  }
                                              },
                                              {
                                                  "order": 6,
                                                  "title": "재계약 우선권 특약",
                                                  "content": "임차인이 계약 만료 2개월 전까지 재계약 의사를 통보하고 월세를 성실히 납부한 경우, 동일 조건으로 재계약할 수 있는 우선권을 갖는다. 임대인은 정당한 사유 없이 재계약을 거부할 수 없다.",
                                                  "assessment": {
                                                      "owner": {
                                                          "level": "주의",
                                                          "reason": "임대인의 임차인 선택권이 제한되나, 성실한 임차인 확보로 안정적인 임대 운영이 가능합니다."
                                                      },
                                                      "tenant": {
                                                          "level": "안심",
                                                          "reason": "주거 안정성이 보장되고 이사 부담이 줄어들어 장기 거주 계획을 세울 수 있습니다."
                                                      }
                                                  }
                                              }
                                          ]
                                      },
                                      "error": None
                                  }
                              }
                          }
                      }
                  }
              },
              400: {
                  "description": "잘못된 요청",
                  "content": {
                      "application/json": {
                          "example": {
                              "success": False,
                              "message": "특약 추천 처리 중 오류가 발생했습니다.",
                              "data": None,
                              "error": "필수 필드가 누락되었습니다."
                          }
                      }
                  }
              }
          })
async def recommend_clauses(request: ClauseRecommendationRequest):
    """특약 추천 API
    
    임대인과 임차인의 사전조사 정보를 분석하여 맞춤형 특약 6개를 생성하고
    각 특약에 대한 양측의 이익 평가를 제공합니다.
    """
    try:
        logger.info("특약 추천 API 호출")
        
        # Request 데이터를 Dict로 변환 (ClauseReportGenerator 입력 형식)
        # by_alias=True를 사용하여 camelCase로 변환
        owner_data = request.owner_data.model_dump(by_alias=True)
        tenant_data = request.tenant_data.model_dump(by_alias=True)
        
        # OCR 데이터는 수동으로 camelCase로 변환
        ocr_data = None
        if request.ocr_data:
            ocr_data = {
                "file_name": request.ocr_data.file_name,
                "extracted_at": request.ocr_data.extracted_at,
                "source": request.ocr_data.source,
                "special_terms": request.ocr_data.special_terms,
                "raw_text": request.ocr_data.raw_text
            }
        
        # 특약 생성 및 평가 프로세스 실행
        result = clause_report_generator.process_clause_generation_request(
            owner_data=owner_data,
            tenant_data=tenant_data,
            ocr_data=ocr_data
        )
        
        # 결과 확인 및 응답
        if result.get("success"):
            logger.info(f"특약 추천 성공: {result.get('data', {}).get('total_clauses', 0)}개 생성")
            return ApiResponse.success(
                data=result.get("data"),
                message=result.get("message", "특약 추천이 완료되었습니다.")
            )
        else:
            logger.error(f"특약 추천 실패: {result.get('message')}")
            return ApiResponse.error(
                message="특약 추천 처리 중 오류가 발생했습니다.",
                code="CLAUSE_GENERATION_ERROR"
            )
            
    except Exception as e:
        logger.error(f"특약 추천 API 오류: {str(e)}")
        logger.error(traceback.format_exc())
        return ApiResponse.error(
            message="특약 추천 중 예상치 못한 오류가 발생했습니다.",
            code="INTERNAL_ERROR"
        )


# 에러 핸들러
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    response = ApiResponse.error(
        message=exc.detail,
        code="HTTP_ERROR"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump(exclude_none=True),
        media_type="application/json; charset=utf-8"
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    response = ApiResponse.error(
        message="Internal server error",
        code="INTERNAL_ERROR"
    )
    return JSONResponse(
        status_code=500,
        content=response.model_dump(exclude_none=True),
        media_type="application/json; charset=utf-8"
    )


if __name__ == "__main__":
    import uvicorn
    
    # UTF-8 환경 보장
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 환경 변수에서 포트 가져오기
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True
    )