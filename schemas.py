from pydantic import BaseModel, Field

class ValidateIDRequest(BaseModel):
    user_id: str = Field(..., example="stu_2290")
    image_base64: str = Field(..., example="<base64_encoded_image>")

class ValidateIDResponse(BaseModel):
    user_id: str
    validation_score: float
    label: str  # genuine, suspicious, fake
    status: str  # approved, manual_review, rejected
    reason: str
    threshold: float
