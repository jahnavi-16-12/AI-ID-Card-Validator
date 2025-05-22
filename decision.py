def decide_label(validation_score, ocr_fields_found, template_match_score, config):
    """
    Combines AI score, OCR fields, and template score to make final decision.
    """
    min_fields = set(config.get("ocr_min_fields", []))
    threshold = config.get("validation_threshold", 0.7)
    approved_colleges = config.get("approved_colleges", [])

    # OCR confidence = how many required fields were detected
    ocr_found = set(ocr_fields_found)
    ocr_confidence = len(ocr_found & min_fields) / len(min_fields) if min_fields else 1.0

    if validation_score > 0.85 and ocr_confidence == 1.0 and template_match_score >= 0.15:
        return "genuine", "approved", "All checks passed"
    elif validation_score < 0.6 or ocr_confidence == 0:
        return "fake", "rejected", "Low AI score or OCR failed"
    else:
        return "suspicious", "manual_review", "Unclear classification or missing fields"
