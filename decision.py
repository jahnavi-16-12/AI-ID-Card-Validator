import json

def decide_label(validation_score, ocr_result, template_match_score, threshold):
    """
    Combines AI score, OCR fields, and template score to make final decision.
    
    Args:
        validation_score: Score from image classifier
        ocr_result: Dictionary containing OCR validation results
        template_match_score: Score from template matching
        threshold: Minimum threshold for validation
    """
    # Calculate confidence based on detected fields
    fields_detected = ocr_result["fields_detected"]
    total_possible_fields = 4  # college, name, roll/class, face
    ocr_confidence = fields_detected / total_possible_fields

    if validation_score > 0.85 and ocr_confidence >= 0.75 and template_match_score >= 0.15:
        return "genuine", "approved", "All checks passed"
    elif validation_score < 0.6 or ocr_confidence < 0.5:
        return "fake", "rejected", "Low AI score or insufficient fields detected"
    else:
        return "suspicious", "manual_review", "Unclear classification or missing fields"
