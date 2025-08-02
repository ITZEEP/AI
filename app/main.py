from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os
import tempfile
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

    class Config:
        populate_by_name = True


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

    class Config:
        populate_by_name = True


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

    class Config:
        populate_by_name = True


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

    class Config:
        populate_by_name = True
        schema_extra = {
            "example": {
                "userId": 123,
                "userType": "tenant",
                "homeId": 456,
                "address": "서울특별시 강남구 테헤란로 123",
                "propertyPrice": 500000000,
                "leaseType": "JEONSE",
                "registryDocument": {
                    "regionAddress": "서울특별시 강남구 대치동 123-45",
                    "roadAddress": "서울특별시 강남구 테헤란로 123",
                    "ownerName": "홍길동",
                    "ownerBirthDate": "1970-01-01",
                    "debtor": "홍길동",
                    "mortgageeList": [
                        {
                            "priorityNumber": 1,
                            "maxClaimAmount": 300000000,
                            "debtor": "홍길동",
                            "mortgagee": "국민은행"
                        }
                    ],
                    "hasSeizure": False,
                    "hasAuction": False,
                    "hasLitigation": False,
                    "hasAttachment": False
                },
                "buildingDocument": {
                    "siteLocation": "서울특별시 강남구 대치동 123-45",
                    "roadAddress": "서울특별시 강남구 테헤란로 123",
                    "totalFloorArea": 84.5,
                    "purpose": "아파트",
                    "floorNumber": 15,
                    "approvalDate": "2010-05-15",
                    "isViolationBuilding": False
                },
                "registeredUserName": "김철수",
                "residenceType": "APARTMENT"
            }
        }


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

    class Config:
        populate_by_name = True
        schema_extra = {
            "example": {
                "contractId": 1,
                "homeId": 100,
                "ownerId": 10,
                "buyerId": 20,
                "contractDate": "2024-01-01",
                "contractExpireDate": "2025-12-31",
                "depositPrice": 200000000,
                "monthlyRent": 1000000,
                "maintenanceFee": 150000,
                "specialClauses": [
                    "임차인은 계약 해지 시 원상복구 비용을 전액 부담한다.",
                    "애완동물 사육을 허가하되, 추가 보증금 50만원을 납부한다.",
                    "임대인은 언제든지 3일 전 통보로 계약을 해지할 수 있다.",
                    "임차인은 전대 및 양도를 할 수 없다.",
                    "월세를 3일 이상 연체 시 연체료는 일 1%로 한다."
                ]
            }
        }


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