import base64
import io
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2

from image_classifier import predict_image_class, load_trained_model
from ocr_validator import extract_ocr_fields
from template_matcher import TemplateMatcher
from decision import decide_label

# ----- Load Config -----
with open("config.json") as f:
    config = json.load(f)

# ----- Load Image Classifier -----
model = load_trained_model()
class_names = ['fake', 'genuine', 'suspicious']

# ----- Initialize Template Matcher -----
matcher = TemplateMatcher(template_dir="test_template/")

# ----- FastAPI Setup -----
app = FastAPI(title="College ID Validator", version="1.0")

# ----- Request Model -----
class ValidationRequest(BaseModel):
    user_id: str
    image_base64: str


# ----- Health Check -----
@app.get("/health")
def health_check():
    return {"status": "ok"}


# ----- Version Check -----
@app.get("/version")
def version_info():
    return {"version": "1.0.0"}


# ----- ID Validation Endpoint -----
@app.post("/validate-id")
def validate_id(request: ValidationRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image_gray = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)

        # --- Step 1: Image Classification ---
        validation_label, confidence = predict_image_class(image_pil, model, class_names)

        # --- Step 2: OCR Field Extraction ---
        ocr_fields = extract_ocr_fields(image_cv)  # should return list like ["name", "roll_number", "college"]

        # --- Step 3: Template Matching ---
        match, template_name, match_score = matcher.is_match(image_gray)

        # --- Step 4: Final Decision ---
        label, status, reason = decide_label(confidence, ocr_fields, match_score, config)

        return {
            "user_id": request.user_id,
            "validation_score": round(confidence, 4),
            "label": label,
            "status": status,
            "reason": reason,
            "threshold": config["validation_threshold"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----- Run with Uvicorn -----
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
