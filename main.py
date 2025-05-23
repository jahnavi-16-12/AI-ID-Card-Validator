import json
import base64
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from schemas import ValidateIDRequest, ValidateIDResponse
import tensorflow as tf
from image_classifier import classify_image
from ocr_validator import validate_id_card  # make sure function name is consistent here
from template_matcher import check_template
from decision import decide_label
from PIL import Image
import io
import uvicorn

# ---------------------------
# Setup logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="College ID Validator")

def load_model():
    try:
        return tf.keras.models.load_model("model/image_model.keras")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise RuntimeError(f"❌ Error loading model: {e}")

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading {path}: {e}")
        raise RuntimeError(f"❌ Error loading {path}: {e}")

# ---------------------------
# Load all dependencies on startup
# ---------------------------
model = load_model()
config = load_json("config.json")
approved_colleges = load_json("approved_colleges.json")

# Load class names for classifier
# For example, you can load them from config or hardcode if fixed classes
class_names = config.get("class_names", ["genuine", "fake", "suspicious"])

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/version")
async def version():
    return {"version": "1.0.0"}

@app.post("/validate-id", response_model=ValidateIDResponse, response_model_exclude_none=True)
async def validate_id(request: ValidateIDRequest, background_tasks: BackgroundTasks):
    # Step 1: Decode base64 image
    try:
        image_bytes = base64.b64decode(request.image_base64)
    except Exception:
        logger.warning("⚠️ Invalid base64 image encoding")
        raise HTTPException(status_code=400, detail="Invalid base64 image encoding")

    # Convert bytes to PIL Image for classify_image function
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        logger.warning("⚠️ Unable to open image from bytes")
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Step 2: Run image classification
    try:
        validation_label, validation_score = classify_image(pil_image, model, class_names)
        logger.info(f"✅ Image classification: {validation_label} ({validation_score:.2f})")
    except Exception as e:
        logger.error(f"❌ Error in image classification: {e}")
        raise HTTPException(status_code=500, detail="Image classification failed")

    # Step 3: Run OCR validation
    try:
        ocr_result = validate_id_card(image_bytes, approved_colleges, config["ocr_min_fields"])
        logger.info("OCR Validation Results:")
        logger.info(f"- College verified: {ocr_result['college_verified']}")
        logger.info(f"- Name verified: {ocr_result['name_verified']}")
        logger.info(f"- Roll/Class verified: {ocr_result['roll_number_verified']}")
        logger.info(f"- Face verified: {ocr_result['face_verified']}")
        logger.info(f"- Fields detected: {ocr_result['fields_detected']}")
        logger.info(f"- OCR text sample: {ocr_result['ocr_text_sample']}")
    except Exception as e:
        logger.error(f"❌ Error in OCR validation: {e}")
        raise HTTPException(status_code=500, detail=f"OCR validation failed: {str(e)}")

    # Step 4: Run template matching
    try:
        template_match = check_template(image_bytes)
        logger.info(f"✅ Template matching: {template_match}")
    except Exception as e:
        logger.error(f"❌ Error in template matching: {e}")
        raise HTTPException(status_code=500, detail="Template matching failed")

    # Step 5: Decide final label
    try:
        label, status, reason = decide_label(
            validation_score,
            ocr_result,
            template_match,
            config["validation_threshold"]
        )
        logger.info(f"✅ Final decision: {label} ({status}) - {reason}")
    except Exception as e:
        logger.error(f"❌ Error in decision logic: {e}")
        raise HTTPException(status_code=500, detail="Decision logic failed")

    # Optional: Offload heavy processing using background tasks
    # background_tasks.add_task(some_async_logging_or_postprocessing, request.user_id, label)

    # Step 6: Return full response
    return ValidateIDResponse(
        user_id=request.user_id,
        validation_score=validation_score,
        label=label,
        status=status,
        reason=reason,
        threshold=config["validation_threshold"]
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
