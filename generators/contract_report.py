"""
generators/contract_report.py - 계약서 적법성 검사 Spring 연동

역할:
1. AI 생성 특약 + 계약서 기본정보 수신 및 파싱
2. 계약서 적법성 검사 수행  
3. 검사 결과를 Spring 형태로 반환
4. 임대차보호법, 민법 위반사항 포맷팅
"""
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.logger_config import get_logger
logger = get_logger(__name__)


class ViolationType(Enum):
    """위반 유형"""
    ILLEGAL = "위반"           # 명백한 법령 위반
    CAUTION = "주의"          # 주의가 필요한 조항
    LEGAL = "적법"            # 법령에 적합


@dataclass
class ContractBasicInfo:
    """계약서 기본 정보 (Spring에서 전달받는 데이터)"""
    contract_chat_id: int
    # 제1조 계약 당사자
    owner_name: str
    owner_addr: str
    owner_phone_num: str
    buyer_name: str
    buyer_addr: str
    buyer_phone_num: str
    # 제2조 임대물건의 표시
    home_addr1: str
    home_addr2: str
    residence_type: str
    exclusive_area: float
    home_floor: int
    # 제3조 임대차 기간 및 임료
    contract_start_date: str
    contract_end_date: str
    deposit_price: int
    monthly_rent: int
    maintenance_fee: int


@dataclass
class GeneratedClause:
    """특약 데이터"""
    order: int
    title: str
    content: str


@dataclass
class ClausesData:
    """AI 생성 특약 전체 데이터"""
    success: bool
    message: str
    timestamp: str
    total_clauses: int
    clauses: List[GeneratedClause]


@dataclass
class LegalViolation:
    """법령 위반 정보"""
    violation_type: ViolationType
    law_name: str
    violation_content: str
    explanation: str
    legal_basis: str
    improvement_example: str
    original_clause: str


@dataclass
class ValidationResult:
    """적법성 검사 결과 (간소화)"""
    success: bool
    contract_chat_id: int
    validation_status: str      # "LEGAL", "CAUTION", "VIOLATION", "ERROR"
    total_violations: int
    violations: List[LegalViolation]
    validated_at: str
    recommendation: str

class ContractDataParser:
    """계약서 데이터 파싱 유틸리티"""
    
    @staticmethod
    def parse_clauses_data(clauses_json: Dict[str, Any]) -> ClausesData:
        """AI 생성 특약 데이터 파싱"""
        try:
            data_section = clauses_json.get('data', {})
            clauses_list = data_section.get('clauses', [])
            
            parsed_clauses = []
            for clause_data in clauses_list:
                parsed_clauses.append(GeneratedClause(
                    order=clause_data.get('order', 0),
                    title=clause_data.get('title', ''),
                    content=clause_data.get('content', '')
                ))
            
            return ClausesData(
                success=clauses_json.get('success', False),
                message=clauses_json.get('message', ''),
                timestamp=clauses_json.get('timestamp', ''),
                total_clauses=data_section.get('total_clauses', len(parsed_clauses)),
                clauses=parsed_clauses
            )
            
        except Exception as e:
            logger.error(f"특약 데이터 파싱 실패: {e}")
            return ClausesData(
                success=False,
                message="파싱 실패",
                timestamp=datetime.now().isoformat(),
                total_clauses=0,
                clauses=[]
            )
    
    @staticmethod
    def parse_contract_basic_info(contract_json: Dict[str, Any]) -> ContractBasicInfo:
        """계약서 기본 정보 파싱"""
        try:
            return ContractBasicInfo(
                contract_chat_id=contract_json.get('contractChatId', 0),
                owner_name=contract_json.get('ownerName', ''),
                owner_addr=contract_json.get('ownerAddr', ''),
                owner_phone_num=contract_json.get('ownerPhoneNum', ''),
                buyer_name=contract_json.get('buyerName', ''),
                buyer_addr=contract_json.get('buyerAddr', ''),
                buyer_phone_num=contract_json.get('buyerPhoneNum', ''),
                home_addr1=contract_json.get('homeAddr1', ''),
                home_addr2=contract_json.get('homeAddr2', ''),
                residence_type=contract_json.get('residenceType', ''),
                exclusive_area=contract_json.get('exclusiveArea', 0.0),
                home_floor=contract_json.get('homeFloor', 0),
                contract_start_date=contract_json.get('contractStartDate', ''),
                contract_end_date=contract_json.get('contractEndDate', ''),
                deposit_price=contract_json.get('depositPrice', 0),
                monthly_rent=contract_json.get('monthlyRent', 0),
                maintenance_fee=contract_json.get('maintenanceFee', 0)
            )
            
        except Exception as e:
            logger.error(f"계약서 기본 정보 파싱 실패: {e}")
            return ContractBasicInfo(
                contract_chat_id=0,
                owner_name="파싱실패", buyer_name="파싱실패",
                owner_addr="", owner_phone_num="", buyer_addr="", buyer_phone_num="",
                home_addr1="", home_addr2="", residence_type="",
                exclusive_area=0.0, home_floor=0,
                contract_start_date="", contract_end_date="",
                deposit_price=0, monthly_rent=0, maintenance_fee=0
            )
    
    @staticmethod
    def format_contract_for_ai_analysis(basic_info: ContractBasicInfo, clauses_data: ClausesData) -> str:
        """AI 분석용 계약서 전체 텍스트 생성"""
        
        contract_parts = []
        
        # 1. 기본 계약 정보
        contract_parts.append("=== 계약서 기본 정보 ===")
        contract_parts.append(f"계약채팅 ID: {basic_info.contract_chat_id}")
        
        # 2. 계약 당사자
        contract_parts.append("=== 제1조 계약 당사자 ===")
        contract_parts.append(f"임대인: {basic_info.owner_name} ({basic_info.owner_addr})")
        contract_parts.append(f"임차인: {basic_info.buyer_name} ({basic_info.buyer_addr})")
        
        # 3. 임대물건 정보
        contract_parts.append("=== 제2조 임대물건의 표시 ===")
        contract_parts.append(f"소재지: {basic_info.home_addr1} {basic_info.home_addr2}")
        contract_parts.append(f"건물유형: {basic_info.residence_type}")
        contract_parts.append(f"전용면적: {basic_info.exclusive_area}㎡")
        contract_parts.append(f"층수: {basic_info.home_floor}층")
        
        # 4. 계약 조건
        contract_parts.append("=== 제3조 임대차 기간 및 임료 ===")
        
        # 날짜 파싱
        try:
            start_date = datetime.strptime(basic_info.contract_start_date, '%Y-%m-%d')
            end_date = datetime.strptime(basic_info.contract_end_date, '%Y-%m-%d')
            period_days = (end_date - start_date).days + 1
            period_years = period_days / 365
            
            contract_parts.append(f"계약기간: {start_date.strftime('%Y년 %m월 %d일')} ~ {end_date.strftime('%Y년 %m월 %d일')} (총 {period_days}일, 약 {period_years:.1f}년)")
        except:
            contract_parts.append(f"계약기간: {basic_info.contract_start_date} ~ {basic_info.contract_end_date}")
        
        # 임대차 유형 및 금액
        if basic_info.monthly_rent == 0:
            # 전세 계약
            contract_parts.append("임대차 유형: 전세 계약")
            contract_parts.append(f"전세보증금: {basic_info.deposit_price:,}원")
            
        else:
            # 월세 계약
            contract_parts.append("임대차 유형: 월세 계약")
            contract_parts.append(f"보증금: {basic_info.deposit_price:,}원")
            contract_parts.append(f"월세: {basic_info.monthly_rent:,}원")
            
            # 월세 비율 계산
            if basic_info.deposit_price > 0:
                monthly_ratio = (basic_info.monthly_rent * 12) / basic_info.deposit_price * 100
                contract_parts.append(f"연간 월세/보증금 비율: {monthly_ratio:.1f}%")
        
        # 관리비
        if basic_info.maintenance_fee > 0:
            contract_parts.append(f"관리비: {basic_info.maintenance_fee:,}원")
        
        # 5. AI 생성 특약사항
        if clauses_data.success and clauses_data.clauses:
            contract_parts.append("=== AI 생성 특약사항 ===")
            for clause in clauses_data.clauses:
                contract_parts.append(f"{clause.order}. [{clause.title}] {clause.content}")
        else:
            contract_parts.append("=== 특약사항 없음 ===")
        
        return "\n".join(contract_parts)


class ContractValidationGenerator:
    """계약서 적법성 검사 결과 생성기"""
    
    @staticmethod
    def validate_contract_with_clauses(clauses_data_json: Dict[str, Any],
                                     contract_basic_info_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI 생성 특약 + 계약서 기본정보로 적법성 검사 후 Spring 형태로 반환
        
        Args:
            clauses_data_json: AI가 생성한 특약 데이터 JSON
            contract_basic_info_json: 계약서 기본 정보 JSON
            
        Returns:
            Spring 형태의 적법성 검사 결과
        """
        try:
            logger.info(f"계약서 적법성 검사 시작 - contractChatId: {contract_basic_info_json.get('contractChatId')}")
            
            # 1. 입력 데이터 파싱
            clauses_data = ContractDataParser.parse_clauses_data(clauses_data_json)
            basic_info = ContractDataParser.parse_contract_basic_info(contract_basic_info_json)
            
            # 2. AI 분석용 계약서 텍스트 생성
            contract_text = ContractDataParser.format_contract_for_ai_analysis(basic_info, clauses_data)
            
            # 3. AI 모델로 적법성 검사
            violations = ContractValidationGenerator._analyze_contract_legality(contract_text, basic_info)
            
            # 4. Spring 응답 형태로 변환
            spring_response = ContractValidationGenerator._convert_to_spring_format(
                violations, basic_info, clauses_data
            )
            
            # 🆕 JSON 예쁘게 출력해서 확인 가능하게!
            print("\n" + "="*80)
            print("📋 Spring으로 보낼 JSON 응답:")
            print("="*80)
            print(json.dumps(spring_response, ensure_ascii=False, indent=2))
            print("="*80 + "\n")
            
            logger.info(f"계약서 적법성 검사 완료 - 총 {len(violations)}건 문제 발견")
            return spring_response
        
        except Exception as e:
            logger.error(f"계약서 적법성 검사 실패: {e}")
            return ContractValidationGenerator._get_error_response()
    
    @staticmethod
    def _analyze_contract_legality(contract_text: str, basic_info: ContractBasicInfo) -> List[LegalViolation]:
        """계약서 적법성 분석 (실제 AI 모델 호출)"""
        try:
            # clause_checker의 간단한 분석 함수 사용
            from model.clause_checker import analyze_contract_text_for_report
            
            # 전세/월세 판단
            is_jeonse = basic_info.monthly_rent == 0
            
            # AI 모델로 분석 (딕셔너리 리스트 반환)
            violation_dicts = analyze_contract_text_for_report(contract_text, is_jeonse)
            
            # 딕셔너리를 LegalViolation 객체로 변환
            violations = []
            for v_dict in violation_dicts:
                violation_type = ViolationType.ILLEGAL if v_dict.get('violation_type') == "위반" else ViolationType.CAUTION
                violations.append(LegalViolation(
                    violation_type=violation_type,
                    law_name=v_dict.get('law_name', ''),
                    violation_content=v_dict.get('violation_content', ''),
                    explanation=v_dict.get('explanation', ''),
                    legal_basis=v_dict.get('legal_basis', ''),
                    improvement_example=v_dict.get('improvement_example', ''),
                    original_clause=v_dict.get('original_clause', '')
                ))
            
            return violations
            
        except Exception as e:
            logger.error(f"AI 적법성 분석 실패: {e}")
            return []
    
    @staticmethod
    def _convert_to_spring_format(violations: List[LegalViolation], 
                                basic_info: ContractBasicInfo,
                                clauses_data: ClausesData) -> Dict[str, Any]:
        """검사 결과를 Spring 형태로 변환 - 위반사항만 간단하게"""
        
        # 전체 상태 판정
        illegal_count = len([v for v in violations if v.violation_type == ViolationType.ILLEGAL])
        caution_count = len([v for v in violations if v.violation_type == ViolationType.CAUTION])
        
        if illegal_count > 0:
            validation_status = "VIOLATION"
        elif caution_count > 0:
            validation_status = "CAUTION"
        else:
            validation_status = "LEGAL"
        
        # 위반사항 리스트만 생성
        violation_details = []
        for violation in violations:
            violation_details.append({
                "violation_type": violation.violation_type.value,
                "law_name": violation.law_name,
                "violation_content": violation.violation_content,
                "explanation": violation.explanation,
                "legal_basis": violation.legal_basis,
                "improvement_example": violation.improvement_example,
                "original_clause": violation.original_clause
            })
        
        
        # 🆕 심플한 Spring 형태 JSON 구조 - 위반사항만!
        return {
            "success": True,
            "contract_chat_id": basic_info.contract_chat_id,
            "validation_status": validation_status,
            "total_violations": len(violations),
            "violations": violation_details,
            "validated_at": datetime.now().isoformat()
        }
    
    
    @staticmethod
    def _get_error_response() -> Dict[str, Any]:
        """오류시 기본 응답"""
        return {
            "success": False,
            "error": "계약서 법령 검토 중 오류가 발생했습니다",
            "contract_chat_id": 0,
            "validation_status": "ERROR",
            "total_violations": 0,
            "violations": [],
            "validated_at": datetime.now().isoformat(),
            "recommendation": "시스템 오류로 인해 법령 검토를 완료할 수 없습니다. 전문가 상담을 권장드립니다."
        }


# 편의 함수
def validate_contract_with_clauses_for_spring(clauses_data: Dict[str, Any], 
                                            contract_basic_info: Dict[str, Any]) -> Dict[str, Any]:
    """Spring용 계약서 적법성 검사 편의 함수"""
    return ContractValidationGenerator.validate_contract_with_clauses(clauses_data, contract_basic_info)


# 사용 예제
if __name__ == "__main__":
    print("\n=== 계약서 적법성 검사 Spring 연동 테스트 ===")
    
    # 테스트용 AI 생성 특약 데이터
    test_clauses_data = {
        "success": True,
        "message": "특약 생성 및 평가 완료",
        "timestamp": "2025-08-04T13:15:02.706857",
        "data": {
            "total_clauses": 3,
            "clauses": [
                {
                    "order": 1,
                    "title": "근저당권 감액 조건부 계약",
                    "content": "임대인은 잔금 지급일까지 본 부동산에 설정된 근저당권을 감액 등기한다."
                },
                {
                    "order": 2,
                    "title": "임대인 임의 해지",
                    "content": "임대인은 언제든지 3일 전 통보로 계약을 해지할 수 있다."  # 위반 조항
                },
                {
                    "order": 3,
                    "title": "전세보증금 반환보증보험",
                    "content": "임대인은 임차인의 전세보증금 반환보증보험 가입에 협조한다."
                }
            ]
        }
    }
    
    # 테스트용 계약서 기본 정보
    test_contract_basic_info = {
        "contractChatId": 1234,
        "ownerName": "홍길동",
        "ownerAddr": "서울특별시 강남구 논현동 123-4",
        "ownerPhoneNum": "01012345678",
        "buyerName": "김영희", 
        "buyerAddr": "서울특별시 마포구 서교동 56-7",
        "buyerPhoneNum": "01098765432",
        "homeAddr1": "서울특별시 용산구 이촌동",
        "homeAddr2": "이촌로 123, 101동 202호",
        "residenceType": "아파트",
        "exclusiveArea": 84.5,
        "homeFloor": 2,
        "contractStartDate": "2025-09-01",
        "contractEndDate": "2027-08-31",
        "depositPrice": 100000000,
        "monthlyRent": 0,  # 전세
        "maintenanceFee": 120000
    }
    
    # 적법성 검사 실행
    print("🔍 계약서 적법성 검사 실행 중...")
    result = validate_contract_with_clauses_for_spring(test_clauses_data, test_contract_basic_info)
    
    print(f"✅ 검토 상태: {result.get('validation_status', 'UNKNOWN')}")
    print(f"📊 총 위반사항: {result.get('total_violations', 0)}건")
    print(f"💡 권고사항: {result.get('recommendation', '정보 없음')}")
    
    print("\n🎉 테스트 완료!")
    print("💡 개선사항:")
    print("   ✅ 심플한 JSON 구조 - 위반사항만!")
    print("   ✅ summary, contract_info 제거로 간소화")
    print("   ✅ Spring에서 필요한 핵심 정보만 전달")
    print("   ✅ 예쁘게 포맷된 JSON 출력으로 확인 가능")