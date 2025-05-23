import pytesseract
import cv2
import numpy as np
import re
from typing import List

# Set up Tesseract path (adjust to your environment)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load OpenCV's Haar Cascade for face detection once (do not reload on every call)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Blacklist of fake or placeholder names to reject
BLACKLIST_NAMES = {
    "mickey mouse", "iron man", "elon musk", "donald duck", "batman",
    "spiderman", "tony stark", "steve jobs", "captain america", "naruto",
    "goku", "superman", "barack obama", "modi", "saitama"
}

def detect_college(text: str, approved_colleges: List[str]) -> bool:
    """Check if any approved college name appears in the OCR text."""
    text_lower = text.lower()
    for college in approved_colleges:
        if college.lower() in text_lower:
            return True
    return False

def detect_name(text: str) -> bool:
    """Detect a valid name in OCR text, ignoring blacklisted names."""
    # Pattern: look for 'Name:' or 'Name -' followed by alphabets and spaces
    name_match = re.search(r'name[:\-]?\s*([A-Za-z .]+)', text, re.IGNORECASE)
    
    if name_match:
        name_found = name_match.group(1).strip().lower()
    else:
        # Fallback heuristic: find lines with 2-3 words of uppercase letters
        # This will match names like "RAM SHRESTHA"
        for line in text.splitlines():
            line = line.strip()
            # Match 2-3 uppercase words (typical name format)
            if re.match(r'^[A-Z][A-Z\s]{2,25}$', line) and len(line.split()) in [2, 3]:
                name_found = line.lower()
                break
        else:
            name_found = None

    if name_found:
        # Clean name by removing any non-alpha characters except space
        name_clean = re.sub(r'[^a-z ]', '', name_found).strip()
        if name_clean in BLACKLIST_NAMES:
            return False
        return True
    return False

def detect_roll_number(text: str) -> bool:
    """Detect roll number or class/section patterns in the OCR text."""
    roll_patterns = [
        # Original college patterns
        r'\b\d{4}-\d{5}[A-Z]{2}\b',   # e.g. 2025-01334CE
        r'\b\d{5}[A-Z]\d{4}\b',       # e.g. 22891A0678
        r'\b\d{2}[A-Z]{2}\d{4}\b',    # e.g. 21CS1001
        r'\b\d{2}[A-Z]{2}\d{6}\b',    # e.g. 19EC123456
        r'\b\d{4}BTECH\d{4}\b',       # e.g. 2019BTECH0001
        r'\b\d{12}\b',                # e.g. 123456789012
        # School patterns
        r'Class\s+\d+',               # e.g. Class 28
        r'Sec\.\s*[A-Z]+',           # e.g. Sec. ZA
        r'\b\d{1,2}(st|nd|rd|th)?\s*(Grade|Class|Std\.?)',  # Various class formats
        r'Section\s*[A-Z]'            # Section format
    ]
    
    text_upper = text.upper()
    # Check for either class or section
    for pattern in roll_patterns:
        if re.search(pattern, text_upper, re.IGNORECASE):
            return True
    return False

def detect_face(image_bytes: bytes) -> bool:
    """Detect face(s) in the image using OpenCV Haar Cascade."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        # Cannot decode image
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

def validate_id_card(image_bytes: bytes, approved_colleges: List[str], min_fields: int) -> dict:
    """
    Validate ID card by checking OCR fields and face detection.

    Args:
        image_bytes: Image in bytes (e.g. decoded base64)
        approved_colleges: List of approved institution names to validate against
        min_fields: Minimum number of required valid fields for ID to be considered valid

    Returns:
        dict: Validation results with boolean flags, count, and OCR sample text.
    """
    # Decode image bytes to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data - cannot decode")

    # Run OCR with Tesseract on the image
    text = pytesseract.image_to_string(img)

    # Run each detection step
    college_found = detect_college(text, approved_colleges)
    name_found = detect_name(text)
    roll_found = detect_roll_number(text)
    face_found = detect_face(image_bytes)

    # Count how many fields are detected as True
    fields_detected = sum([college_found, name_found, roll_found, face_found])

    # Check if detected fields meet the minimum required
    is_valid = fields_detected >= min_fields

    # Return detailed validation dictionary
    return {
        "college_verified": college_found,
        "name_verified": name_found,
        "roll_number_verified": roll_found,
        "face_verified": face_found,
        "fields_detected": fields_detected,
        "is_valid": is_valid,
        "ocr_text_sample": text[:200]  # first 200 chars for debugging/logging
    }