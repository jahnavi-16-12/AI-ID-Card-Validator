from pydantic import BaseModel, Field

class ValidateIDRequest(BaseModel):
    user_id: str = Field(..., json_schema_extra={"example": "stu_2290"})
    image_base64: str = Field(..., json_schema_extra={"example": "<base64_encoded_image>"})

class ValidateIDResponse(BaseModel):
    user_id: str
    validation_score: float
    label: str  # genuine, suspicious, fake
    status: str  # approved, manual_review, rejected
    reason: str
    threshold: float
    ocr_fields_detected: int
    ocr_confidence: float
    ocr_text_sample: str
