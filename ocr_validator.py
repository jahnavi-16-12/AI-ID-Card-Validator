import pytesseract
import cv2
import numpy as np
import re
import logging
from typing import List

logger = logging.getLogger(__name__)

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
    logger.info("Checking for college name in text:")
    logger.info(f"Text sample: {text[:200]}")
    
    # First check exact matches from approved list
    for college in approved_colleges:
        if college.lower() in text_lower:
            logger.info(f"Found approved college: {college}")
            return True
    
    # Common educational institution keywords
    edu_keywords = [
        "college", "university", "institute", "school",
        "education", "polytechnic", "academy"
    ]
    
    # Check for educational keywords
    for keyword in edu_keywords:
        if keyword in text_lower:
            # Look for full institution name pattern
            lines = text.splitlines()
            for line in lines:
                if keyword in line.lower() and len(line.strip()) > len(keyword):
                    logger.info(f"Found educational institution: {line.strip()}")
                    return True
    
    logger.info("No approved college or educational institution found in text")
    return False

def detect_name(text: str) -> bool:
    """Detect a valid name in OCR text, ignoring blacklisted names."""
    logger.info("Looking for name in text")
    
    # Multiple name patterns to try
    name_patterns = [
        r'name[:\-]?\s*([A-Za-z\. ]+)',  # Name: John Doe
        r'(?:mr|mrs|ms|dr)[.\s]+([A-Za-z\. ]+)',  # Mr. John Doe
        r'student[:\-]?\s*([A-Za-z\. ]+)',  # Student: John Doe
        r'^([A-Z][A-Za-z\. ]+(?:\s+[A-Z][A-Za-z\. ]+){1,3})$',  # JOHN DOE
    ]
    
    # Try each pattern
    for pattern in name_patterns:
        for line in text.splitlines():
            match = re.search(pattern, line.strip(), re.IGNORECASE)
            if match:
                name_found = match.group(1).strip().lower()
                logger.info(f"Found name with pattern '{pattern}': {name_found}")
                
                # Clean name and check blacklist
                name_clean = re.sub(r'[^a-z\. ]', '', name_found).strip()
                if name_clean in BLACKLIST_NAMES:
                    logger.info(f"Name '{name_clean}' found in blacklist")
                    continue
                    
                if len(name_clean.replace('.', '').replace(' ', '')) >= 3:
                    logger.info(f"Valid name found: {name_clean}")
                    return True
                
    logger.info("No valid name pattern found")
    return False

def detect_roll_number(text: str) -> bool:
    """Detect roll number or class/section patterns in the OCR text."""
    logger.info("Looking for roll number/class patterns")
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
        r'Section\s*[A-Z]',           # Section format
        # Course patterns
        r'Course\s*(ID|Number|No\.?)?\s*[:=]?\s*[\w\d-]+',  # Course ID/Number
        r'Certificate\s*(ID|Number|No\.?)?\s*[:=]?\s*[\w\d-]+',  # Certificate number
        # Additional patterns
        r'Roll\s*[Nn]o\.?\s*[:=]?\s*\d+',  # Roll No: 123
        r'Admission\s*[Nn]o\.?\s*[:=]?\s*[\w\d/-]+',  # Admission No: ABC123
        r'Registration\s*[Nn]o\.?\s*[:=]?\s*[\w\d/-]+',  # Registration No: XYZ789
        r'Student\s*(ID|Number|No\.?)?\s*[:=]?\s*[\w\d/-]+',  # Student ID
        r'\b[A-Z]{2,4}\d{3,8}\b',  # Generic alphanumeric ID pattern
        r'Session\s*[:=]?\s*\d{4}[-/]\d{2,4}'  # Session: 2013-2014
    ]
    
    text_upper = text.upper()
    logger.info("Text sample for roll number search:")
    logger.info(text_upper[:200])
    
    # Check for either class or section
    for pattern in roll_patterns:
        if re.search(pattern, text_upper, re.IGNORECASE):
            logger.info(f"Found matching pattern: {pattern}")
            return True
    logger.info("No roll number/class patterns found")
    return False

def detect_face(image_bytes: bytes) -> bool:
    """Detect face(s) in the image using OpenCV Haar Cascade."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Cannot decode image for face detection")
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        logger.info(f"Found {len(faces)} face(s) in image")
        return True
    logger.info("No faces detected in image")
    return False

def validate_id_card(image_bytes: bytes, approved_colleges: List[str], min_fields: int) -> dict:
    """
    Validate ID card by checking OCR fields and face detection.
    
    For a genuine college ID card, we expect:
    - College name from approved list
    - Student name
    - Roll number/registration number
    - Face detection
    
    Missing or unrecognized fields will lower the confidence score.

    Args:
        image_bytes: Image in bytes (e.g. decoded base64)
        approved_colleges: List of approved institution names to validate against
        min_fields: Minimum number of required valid fields for ID to be considered valid

    Returns:
        dict: Validation results with boolean flags, count, and OCR sample text.
    """
    logger.info("\n=== Starting ID Card Validation ===")
    
    # Decode image bytes to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data - cannot decode")

    # Run OCR with Tesseract on the image
    text = pytesseract.image_to_string(img)
    logger.info("OCR Text extracted:")
    logger.info("-------------------")
    logger.info(text[:500])
    logger.info("-------------------")

    # Run each detection step
    college_found = detect_college(text, approved_colleges)
    name_found = detect_name(text)
    roll_found = detect_roll_number(text)
    face_found = detect_face(image_bytes)

    # Count how many fields are detected as True
    fields_detected = sum([college_found, name_found, roll_found, face_found])
    
    # Log detailed validation results
    logger.info("\n=== Validation Results ===")
    logger.info(f"College found: {college_found}")
    logger.info(f"Name found: {name_found}")
    logger.info(f"Roll/Class found: {roll_found}")
    logger.info(f"Face found: {face_found}")
    logger.info(f"Total fields detected: {fields_detected}/{4}")

    # For college IDs, we expect all these fields
    # Missing college name or roll number indicates this might not be a college ID
    if not college_found or not roll_found:
        logger.info("Missing critical college ID fields (college name or roll number)")
        fields_detected = min(fields_detected, 2)  # Cap at 2 if missing critical fields

    # Check if detected fields meet the minimum required
    is_valid = fields_detected >= min_fields
    logger.info(f"Validation {'passed' if is_valid else 'failed'} (min fields: {min_fields})")

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