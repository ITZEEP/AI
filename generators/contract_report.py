"""
generators/contract_report.py - 계약서 법령 적법성 검토 Spring 연동

역할:
1. Spring에서 계약서 데이터 수신 (ERD 기반)
2. AI 모델로 법령 적법성 검토 수행  
3. 검토 결과를 Spring 형태로 반환
4. 이미지에서 보인 형태로 위반사항 포맷팅
"""
import sys
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dateutil import parser as date_parser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 내부 모듈 - 기존 LLM 코드에 맞게 import
from model.clause_checker import (
    ContractInfo, 
    LegalViolation, 
    ViolationType,
    check_contract_legality_for_spring
)

logger = logging.getLogger(__name__)


class ContractValidationGenerator:
    """계약서 법령 검토 결과 생성기 - Spring 연동 전용"""
    
    def __init__(self):
        """계약서 검토 모델 초기화 - 기존 LLM 코드 활용"""
        # 기존 clause_checker.py의 편의 함수 활용
        logger.info("ContractValidationGenerator 초기화 완료")
    
    def validate_contract_for_spring(self, 
                                   contract_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Spring에서 받은 계약서 데이터로 법령 검토 후 Spring 형태로 반환
        
        Args:
            contract_data: Spring에서 전달된 계약서 데이터 (ERD 기반)
            {
                "contract_id": 1,
                "home_id": 1,
                "owner_id": 1,
                "buyer_id": 2,
                "contract_date": "2024-01-01T00:00:00",
                "contract_expire_date": "2025-12-31T00:00:00",
                "deposit_price": 150000000,
                "monthly_rent": 500000,
                "maintenance_fee": 100000,
                "special_clauses": [
                    "임차인은 계약 해지 시 원상복구 비용을 전액 부담한다.",
                    "애완동물 사육을 허가하되, 추가 보증금 50만원을 납부한다."
                ]
            }
            
        Returns:
            Spring 형태의 검토 결과
        """
        try:
            logger.info(f"계약서 법령 검토 시작 - contract_id: {contract_data.get('contract_id')}")
            
            # 1. Spring 데이터를 AI 모델 형태로 변환
            contract_info = self._parse_spring_contract_data(contract_data)
            
            # 2. 기존 LLM의 Spring 연동 편의 함수 활용
            violations = check_contract_legality_for_spring(
                contract_id=contract_info.contract_id,
                home_id=contract_info.home_id,
                owner_id=contract_info.owner_id,
                buyer_id=contract_info.buyer_id,
                contract_date=contract_info.contract_date,
                contract_expire_date=contract_info.contract_expire_date,
                special_clauses=contract_info.special_clauses,
                deposit_price=contract_info.deposit_price,
                monthly_rent=contract_info.monthly_rent,
                maintenance_fee=contract_info.maintenance_fee
            )
            
            # 3. Spring 형태로 결과 변환
            spring_response = self._convert_to_spring_format(violations, contract_info)
            
            logger.info(f"계약서 검토 완료 - 총 {len(violations)}건 문제 발견")
            return spring_response
        
        except Exception as e:
            logger.error(f"계약서 검토 실패: {e}")
            return self._get_spring_fallback_response()
    
    def _parse_spring_contract_data(self, spring_data: Dict[str, Any]) -> ContractInfo:
        """Spring 계약서 데이터를 AI 모델 형태로 변환"""
        try:
            # 날짜 문자열을 datetime 객체로 변환
            contract_date = self._parse_datetime(spring_data.get('contract_date'))
            contract_expire_date = self._parse_datetime(spring_data.get('contract_expire_date'))
            
            return ContractInfo(
                contract_id=spring_data.get('contract_id', 0),
                home_id=spring_data.get('home_id', 0),
                owner_id=spring_data.get('owner_id', 0),
                buyer_id=spring_data.get('buyer_id', 0),
                contract_date=contract_date,
                contract_expire_date=contract_expire_date,
                deposit_price=spring_data.get('deposit_price'),
                monthly_rent=spring_data.get('monthly_rent'),
                maintenance_fee=spring_data.get('maintenance_fee'),
                special_clauses=spring_data.get('special_clauses', [])
            )
            
        except Exception as e:
            logger.error(f"Spring 데이터 파싱 실패: {e}")
            # 기본값으로 생성
            return ContractInfo(
                contract_id=0,
                home_id=0,
                owner_id=0,
                buyer_id=0,
                contract_date=datetime.now(),
                contract_expire_date=datetime.now(),
                special_clauses=[]
            )
    
    def _parse_datetime(self, date_str: Optional[str]) -> datetime:
        """날짜 문자열을 datetime 객체로 변환"""
        if not date_str:
            raise ValueError("날짜 문자열이 비어있습니다")
        
        try:
            return date_parser.parse(date_str)
        except Exception as e:
            logger.warning(f"날짜 파싱 실패: {date_str}, 오류: {e}")
            raise ValueError(f"날짜 파싱 실패: {date_str}")
    
    def _convert_to_spring_format(self, violations: List[LegalViolation], 
                                 contract_info: ContractInfo) -> Dict[str, Any]:
        """AI 분석 결과를 Spring 형태로 변환"""
        
        # 위반사항들을 Spring DetailGroup 형태로 변환
        violation_details = []
        
        for violation in violations:
            # 이미지에서 본 형태로 포맷팅
            violation_detail = {
                "violation_type": violation.violation_type.value,  # "위반", "주의", "적법"
                "law_name": violation.law_name,                   # "주택임대차보호법"
                "violation_content": violation.violation_content, # "잘못된 내용"
                "explanation": violation.explanation,             # "내용에 대한 설명"
                "legal_basis": violation.legal_basis,             # "제6조 제1항"
                "improvement_example": violation.improvement_example, # "개선 방안 예시"
                "original_clause": violation.original_clause      # "원본 조항"
            }
            violation_details.append(violation_detail)
        
        # 전체 계약서 상태 판정
        overall_status = self._determine_overall_status(violations)
        
        return {
            "success": True,
            "contract_id": contract_info.contract_id,
            "validation_status": overall_status,
            "total_violations": len(violations),
            "violation_summary": {
                "illegal_count": len([v for v in violations if v.violation_type == ViolationType.ILLEGAL]),
                "caution_count": len([v for v in violations if v.violation_type == ViolationType.CAUTION])
            },
            "violations": violation_details,
            "validated_at": datetime.now().isoformat(),
            "recommendation": self._get_overall_recommendation(violations)
        }
    
    def _determine_overall_status(self, violations: List[LegalViolation]) -> str:
        """전체 계약서 상태 판정"""
        if not violations:
            return "LEGAL"  # 적법
        
        # 명백한 위반이 있는 경우
        if any(v.violation_type == ViolationType.ILLEGAL for v in violations):
            return "VIOLATION"  # 위반
        
        # 주의사항만 있는 경우
        return "CAUTION"  # 주의
    
    def _get_overall_recommendation(self, violations: List[LegalViolation]) -> str:
        """전체적인 권고사항 생성"""
        if not violations:
            return "검토 결과 법령에 위반되는 조항이 발견되지 않았습니다."
        
        illegal_count = len([v for v in violations if v.violation_type == ViolationType.ILLEGAL])
        caution_count = len([v for v in violations if v.violation_type == ViolationType.CAUTION])
        
        if illegal_count > 0:
            return f"법령 위반 조항 {illegal_count}건이 발견되어 계약서 수정이 필요합니다. 전문가 상담을 권장드립니다."
        elif caution_count > 0:
            return f"주의가 필요한 조항 {caution_count}건이 발견되었습니다. 해당 조항들을 검토해보시기 바랍니다."
        else:
            return "계약서 검토가 완료되었습니다."
    
    def _get_spring_fallback_response(self) -> Dict[str, Any]:
        """오류시 Spring 기본 응답"""
        return {
            "success": False,
            "error": "계약서 법령 검토 중 오류가 발생했습니다",
            "contract_id": 0,
            "validation_status": "ERROR",
            "total_violations": 0,
            "violation_summary": {
                "illegal_count": 0,
                "caution_count": 0
            },
            "violations": [],
            "validated_at": datetime.now().isoformat(),
            "recommendation": "시스템 오류로 인해 법령 검토를 완료할 수 없습니다. 전문가 상담을 권장드립니다."
        }


class ContractClauseParser:
    """계약서 조항 파싱 유틸리티"""
    
    @staticmethod
    def parse_contract_from_pdf_data(pdf_extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """PDF 추출 데이터를 계약서 정보로 변환"""
        try:
            # PDF에서 추출된 특약사항 파싱
            special_clauses = pdf_extracted_data.get('special_terms', [])
            
            # 기본 계약 정보 추출 (PDF OCR 결과에서)
            contract_info = {
                "special_clauses": special_clauses,
                "contract_date": pdf_extracted_data.get('contract_date'),
                "contract_expire_date": pdf_extracted_data.get('contract_expire_date'),
                "deposit_price": pdf_extracted_data.get('deposit_price'),
                "monthly_rent": pdf_extracted_data.get('monthly_rent'),
                "maintenance_fee": pdf_extracted_data.get('maintenance_fee')
            }
            
            return contract_info
            
        except Exception as e:
            logger.error(f"PDF 데이터 파싱 실패: {e}")
            return {"special_clauses": []}
    
    @staticmethod
    def extract_clause_categories(clauses: List[str]) -> Dict[str, List[str]]:
        """특약 조항들을 카테고리별로 분류"""
        categories = {
            "임대료_관련": [],
            "계약기간_관련": [],
            "시설_관리": [],
            "생활_규칙": [],
            "기타": []
        }
        
        for clause in clauses:
            if any(keyword in clause for keyword in ["임대료", "보증금", "월세", "관리비"]):
                categories["임대료_관련"].append(clause)
            elif any(keyword in clause for keyword in ["계약기간", "갱신", "해지", "연장"]):
                categories["계약기간_관련"].append(clause)
            elif any(keyword in clause for keyword in ["수리", "보수", "시설", "설비"]):
                categories["시설_관리"].append(clause)
            elif any(keyword in clause for keyword in ["소음", "흡연", "애완동물", "반려동물"]):
                categories["생활_규칙"].append(clause)
            else:
                categories["기타"].append(clause)
        
        return categories


# 편의 함수
def validate_contract_for_spring(contract_data: Dict[str, Any]) -> Dict[str, Any]:
    """Spring용 계약서 법령 검토 편의 함수 - 기존 LLM 활용"""
    generator = ContractValidationGenerator()
    return generator.validate_contract_for_spring(contract_data)


def parse_pdf_contract_data(pdf_data: Dict[str, Any]) -> Dict[str, Any]:
    """PDF 추출 데이터 파싱 편의 함수"""
    return ContractClauseParser.parse_contract_from_pdf_data(pdf_data)


# 사용 예제
if __name__ == "__main__":
    print("\n=== 계약서 법령 검토 Spring 연동 테스트 (기존 LLM 활용) ===")
    
    # 테스트용 Spring 계약서 데이터 (ERD 기반)
    test_spring_data = {
        "contract_id": 1,
        "home_id": 1,
        "owner_id": 1,
        "buyer_id": 2,
        "contract_date": "2024-01-01T00:00:00",
        "contract_expire_date": "2025-12-31T00:00:00",
        "deposit_price": 150000000,  # 1.5억원
        "monthly_rent": 500000,
        "maintenance_fee": 100000,
        "special_clauses": [
            "임차인은 계약 해지 시 원상복구 비용을 전액 부담한다.",
            "애완동물 사육을 허가하되, 추가 보증금 50만원을 납부한다.",
            "임대인은 언제든지 3일 전 통보로 계약을 해지할 수 있다.",  # 명백한 위반
            "임차인은 전대 및 양도를 할 수 없다."
        ]
    }
    
    # 상세 법령 검토
    print("🔍 상세 법령 검토 실행 중...")
    detailed_result = validate_contract_for_spring(test_spring_data)
    
    print(f"검토 상태: {detailed_result['validation_status']}")
    print(f"총 위반사항: {detailed_result['total_violations']}건")
    print(f"권고사항: {detailed_result['recommendation']}")
    
    if detailed_result['violations']:
        print("\n📋 발견된 문제점들:")
        for i, violation in enumerate(detailed_result['violations'], 1):
            print(f"\n--- {i}번째 문제점 ---")
            print(f"어느 법령: {violation['law_name']}")
            print(f"잘못된 내용: {violation['violation_content']}")
            print(f"내용에 대한 설명: {violation['explanation']}")
            print(f"근거: {violation['legal_basis']}")
            print(f"🔧 개선방안: {violation['improvement_example']}")
    
    print("\n🎉 테스트 완료!")
    print("\n💡 Spring 연동 방법 (기존 LLM 코드 활용):")
    print("   1. POST /api/contract/validate")
    print("   2. Request Body: contract_data (ERD 기반)")
    print("   3. Response: validation 결과 (기존 clause_checker.py LLM 처리)")
    print("   4. LLM 모델: model/clause_checker.py의 ContractLegalChecker 활용")