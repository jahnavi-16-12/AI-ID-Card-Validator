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
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="College ID Validator")

def load_model():
    try:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model", "image_model.onnx"))
        session = ort.InferenceSession(model_path)
        input_details = session.get_inputs()[0]
        logger.info(f"✅ ONNX Model loaded successfully")
        logger.info(f"Model input name: {input_details.name}")
        logger.info(f"Model input shape: {input_details.shape}")
        logger.info(f"Model input type: {input_details.type}")
        logger.info(f"Model input details: {input_details}")
        return session
    except Exception as e:
        logger.error(f"❌ Error loading ONNX model: {e}")
        raise RuntimeError(f"❌ Error loading ONNX model: {e}")

def load_json(path):
    try:
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), path)), "r") as f:
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
    """
    Preprocess image for ONNX model input:
    1. Resize to 224x224
    2. Convert to float32 and normalize using ImageNet stats
    3. Convert to NCHW format (batch, channels, height, width)
    """
    logger.info("Starting image preprocessing...")
    logger.info(f"Input image mode: {pil_image.mode}")
    logger.info(f"Input image size: {pil_image.size}")
    
    # Convert to RGB if not already
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
        logger.info("Converted image to RGB mode")
    
    # Resize with antialiasing for better quality
    img = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
    logger.info(f"After resize: {img.size}")
    
    # Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32)
    img_array = img_array / 255.0  # Scale to [0,1]
    
    # ImageNet normalization (ensure float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    logger.info(f"After numpy conversion: shape={img_array.shape}, dtype={img_array.dtype}")
    logger.info(f"Array value range: min={img_array.min():.3f}, max={img_array.max():.3f}")
    
    # Ensure shape is (224, 224, 3)
    if img_array.shape != (224, 224, 3):
        logger.warning(f"Invalid image shape after preprocessing: {img_array.shape}")
        raise ValueError(f"Invalid image shape after preprocessing: {img_array.shape}")
    
    # Convert from NHWC to NCHW format
    img_array = np.transpose(img_array, (2, 0, 1))  # (H,W,C) -> (C,H,W)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension -> (1,C,H,W)
    
    # Ensure float32 type
    img_array = img_array.astype(np.float32)
    
    logger.info(f"Final preprocessed shape: {img_array.shape}")
    logger.info(f"Final array stats: min={img_array.min():.3f}, max={img_array.max():.3f}, mean={img_array.mean():.3f}")
    return img_array

def classify_image_onnx(pil_image, session, class_names):
    try:
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        logger.info(f"Model expects input shape: {input_shape}")
        logger.info(f"Model input name: {input_name}")
        
        img_array = preprocess_image(pil_image)
        logger.info(f"Preprocessed image shape: {img_array.shape}")
        logger.info(f"Input array stats: min={img_array.min():.3f}, max={img_array.max():.3f}, mean={img_array.mean():.3f}")
        
        outputs = session.run(None, {input_name: img_array})
        logger.info(f"Raw model output shape: {[out.shape for out in outputs]}")
        logger.info(f"Raw output values: {outputs[0]}")
        
        scores = outputs[0][0]  # (1, num_classes) -> [num_classes]
        logger.info(f"Raw scores: {scores}")
        
        exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
        logger.info(f"Exp scores: {exp_scores}")
        
        probs = exp_scores / exp_scores.sum()
        logger.info(f"Class probabilities: {dict(zip(class_names, probs))}")
        logger.info(f"Class probabilities array: {probs}")
        
        pred_idx = np.argmax(probs)
        logger.info(f"Predicted class: {class_names[pred_idx]} (index {pred_idx})")
        logger.info(f"Confidence: {float(probs[pred_idx]):.3f}")
        
        return class_names[pred_idx], float(probs[pred_idx])
    except Exception as e:
        logger.error(f"❌ Exception in classify_image_onnx: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
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
        logger.info(f"Decision inputs:")
        logger.info(f"- Validation score: {validation_score:.3f}")
        logger.info(f"- OCR fields detected: {ocr_result['fields_detected']}/{4}")
        logger.info(f"- OCR confidence: {ocr_result['fields_detected']/4:.3f}")
        logger.info(f"- Template match score: {template_match}")
        logger.info(f"- Threshold: {config['validation_threshold']}")
        
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
