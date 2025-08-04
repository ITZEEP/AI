from typing import Optional, Any
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field, ConfigDict


class ErrorDetails(BaseModel):
    """에러 상세 정보"""
    code: Optional[str] = None
    field: Optional[str] = None
    rejected_value: Optional[Any] = Field(None, alias="rejectedValue")
    reason: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class ApiResponse(BaseModel):
    """
    백엔드와 동일한 형식의 API 응답 모델
    """
    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[ErrorDetails] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone(timedelta(hours=9))))

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={
            # Java LocalDateTime 형식: yyyy-MM-dd'T'HH:mm:ss
            datetime: lambda v: v.strftime('%Y-%m-%dT%H:%M:%S')
        }
    )


def create_success_response(
    data: Optional[Any] = None,
    message: Optional[str] = None
) -> ApiResponse:
    """성공 응답 생성"""
    return ApiResponse(
        success=True,
        data=data,
        message=message
    )


def create_error_response(
    message: str,
    code: Optional[str] = None,
    field: Optional[str] = None,
    rejected_value: Optional[Any] = None,
    reason: Optional[str] = None
) -> ApiResponse:
    """에러 응답 생성"""
    error_details = None
    if any([code, field, rejected_value, reason]):
        error_details = ErrorDetails(
            code=code,
            field=field,
            rejected_value=rejected_value,
            reason=reason or message
        )
    
    return ApiResponse(
        success=False,
        message=message,
        error=error_details
    )


# For backward compatibility
ApiResponse.success = staticmethod(create_success_response)
ApiResponse.error = staticmethod(create_error_response)