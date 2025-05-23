import json
import base64
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from schemas import ValidateIDRequest, ValidateIDResponse
import onnxruntime as ort
from ocr_validator import validate_id_card
from template_matcher import check_template
from decision import decide_label
from PIL import Image
import io
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="College ID Validator")

def load_model():
    try:
        session = ort.InferenceSession("model/image_model.onnx")
        logger.info(f"✅ ONNX Model loaded successfully")
        logger.info(f"Model input shape: {session.get_inputs()[0].shape}")
        logger.info(f"Model input type: {session.get_inputs()[0].type}")
        return session
    except Exception as e:
        logger.error(f"❌ Error loading ONNX model: {e}")
        raise RuntimeError(f"❌ Error loading ONNX model: {e}")

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading {path}: {e}")
        raise RuntimeError(f"❌ Error loading {path}: {e}")

# Load model and configs
model_session = load_model()
config = load_json("config.json")
approved_colleges = load_json("approved_colleges.json")
class_names = config.get("class_names", ["genuine", "fake", "suspicious"])

def preprocess_image(pil_image):
    img = pil_image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    if img_array.ndim == 2:  # grayscale to RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    img_array = np.transpose(img_array, (0, 3, 1, 2))  # Change to (1, 3, 224, 224) for ONNX
    return img_array

def classify_image_onnx(pil_image, session, class_names):
    try:
        input_name = session.get_inputs()[0].name
        img_array = preprocess_image(pil_image)
        logger.debug(f"Image array shape for ONNX model: {img_array.shape}")
        outputs = session.run(None, {input_name: img_array})
        logger.debug(f"Raw ONNX output: {outputs}")
        scores = outputs[0][0]  # (1, num_classes) -> [num_classes]
        exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
        probs = exp_scores / exp_scores.sum()
        pred_idx = np.argmax(probs)
        return class_names[pred_idx], float(probs[pred_idx])
    except Exception as e:
        logger.error(f"❌ Exception in classify_image_onnx: {e}")
        raise

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/version")
async def version():
    return {"version": "1.0.0"}

@app.post("/validate-id", response_model=ValidateIDResponse, response_model_exclude_none=True)
async def validate_id(request: ValidateIDRequest, background_tasks: BackgroundTasks):
    try:
        image_bytes = base64.b64decode(request.image_base64)
    except Exception:
        logger.warning("⚠️ Invalid base64 image encoding")
        raise HTTPException(status_code=400, detail="Invalid base64 image encoding")

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        logger.warning("⚠️ Unable to open image from bytes")
        raise HTTPException(status_code=400, detail="Invalid image data")

    try:
        validation_label, validation_score = classify_image_onnx(pil_image, model_session, class_names)
        logger.info(f"✅ Image classification: {validation_label} ({validation_score:.2f})")
    except Exception as e:
        logger.error(f"❌ Error in image classification: {e}")
        raise HTTPException(status_code=500, detail="Image classification failed")

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

    try:
        template_match = check_template(image_bytes)
        logger.info(f"✅ Template matching: {template_match}")
    except Exception as e:
        logger.error(f"❌ Error in template matching: {e}")
        raise HTTPException(status_code=500, detail="Template matching failed")

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

    return ValidateIDResponse(
        user_id=request.user_id,
        validation_score=validation_score,
        label=label,
        status=status,
        reason=reason,
        threshold=config["validation_threshold"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
