import json
import logging

logger = logging.getLogger(__name__)

def decide_label(validation_score, ocr_result, template_match_score, threshold):
    """
    Combines AI score, OCR fields, and template matching to make final decision.
    
    Requirements (exactly as specified):
    - High score > 0.85           -> genuine/approved
    - Score 0.6-0.85 OR          -> suspicious/manual_review
      low OCR confidence
    - Score < 0.6 OR             -> fake/rejected
      OCR fails completely
    
    Args:
        validation_score: Score from image classifier (0.0 to 1.0)
        ocr_result: Dictionary containing OCR validation results
        template_match_score: Template matching score (0.0 to 1.0)
        threshold: Minimum threshold for validation
    """
    logger.info("\n=== Decision Process ===")
    logger.info(f"Validation score: {validation_score:.3f}")
    logger.info(f"Template match score: {template_match_score:.3f}")
    
    # Calculate OCR confidence
    fields_detected = ocr_result["fields_detected"]
    total_possible_fields = 4  # college, name, roll/class, face
    
    # Cap fields at 2 for non-college IDs (e.g., certificates)
    if "certificate" in ocr_result.get("type", "").lower():
        total_possible_fields = 2  # Only require name and ID fields
        
    ocr_confidence = fields_detected / total_possible_fields
    logger.info(f"OCR fields detected: {fields_detected}/{total_possible_fields}")
    logger.info(f"OCR confidence: {ocr_confidence:.3f}")
    
    # OCR failure = less than 50% fields detected
    ocr_failed = ocr_confidence < 0.5
    # Low OCR confidence = less than 75% fields detected
    low_ocr_confidence = ocr_confidence < 0.75
    
    logger.info(f"OCR failed (< 50%): {ocr_failed}")
    logger.info(f"Low OCR confidence (< 75%): {low_ocr_confidence}")
    
    # Combine scores with adjusted weights
    combined_score = (validation_score * 0.5 + 
                     ocr_confidence * 0.3 + 
                     template_match_score * 0.2)
    logger.info(f"Combined score: {combined_score:.3f}")

    # Decision logic with relaxed thresholds
    if validation_score > 0.85 and ocr_confidence >= 0.75:
        logger.info("Decision: GENUINE (high validation score and good OCR)")
        return "genuine", "approved", "High confidence in validation and OCR"
    elif validation_score < 0.4 or (ocr_failed and template_match_score < 0.3):
        logger.info(f"Decision: FAKE (very low scores or multiple failures)")
        logger.info(f"Reason: score < 0.4 = {validation_score < 0.4}, OCR failed = {ocr_failed}, template_match < 0.3 = {template_match_score < 0.3}")
        return "fake", "rejected", "Very low confidence scores or multiple validation failures"
    else:
        logger.info("Decision: SUSPICIOUS (medium confidence or mixed results)")
        return "suspicious", "manual_review", "Medium confidence scores or inconsistent validation results"
