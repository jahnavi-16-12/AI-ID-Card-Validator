import pytesseract
import cv2
import re
import os

# Set up Tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Folder containing test images
folder_path = "test_images"


# College keywords
college_keywords = [
    "vignan", "jntuh", "osmania", "iiit", "vit", "srm", "amrita", "bits pilani",
    "iit", "nit", "kakatiya", "klu", "gokaraju", "muffakham jah", "cbit",
    "vasavi", "geethanjali", "cmr", "mallareddy", "sreenidhi","school inc"
]

# Fake name blacklist
blacklist_names = [
    "mickey mouse", "iron man", "elon musk", "donald duck", "batman",
    "spiderman", "tony stark", "steve jobs", "captain america", "naruto",
    "goku", "superman", "barack obama", "modi", "saitama"
]

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Loop through each image in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Unable to read image: {filename}")
            continue

        print(f"\n--- Checking Image: {filename} ---")

        # OCR text extraction
        text = pytesseract.image_to_string(image)
        text_lower = text.lower()

        # College matching
        college_match = any(college in text_lower for college in college_keywords)
        college_status = "✅ College is Verified" if college_match else "❌ College Not Detected"

        # Name detection
        name_match = re.search(r'(name[:\-]?\s*)([A-Za-z .]+)', text, re.IGNORECASE)
        if name_match:
            name_found = name_match.group(2).strip().lower()
        else:
            lines = text.split('\n')
            candidate_names = []
            for line in lines:
                line = line.strip()
                if re.match(r'^([A-Z]\.\s)?([A-Z]{3,}\s?)+$', line):
                    candidate_names.append(line)
            name_found = candidate_names[0].strip().lower() if candidate_names else None

        if name_found:
            name_clean = re.sub(r'[^a-z ]', '', name_found).strip()
            if name_clean in blacklist_names:
                name_status = f"❌ Name Field: {name_clean.title()} (Blacklisted Fake Name)"
            else:
                name_status = f"✅ Name Field: {name_clean.title()} (Potentially Valid Name)"
        else:
            name_status = "❌ Name Field Not Detected"
        
        # Roll number detection with multiple patterns
        patterns = [
            r'\b\d{4}-\d{5}[A-Z]{2}\b',     # e.g. 2025-01334CE
            r'\b\d{5}[A-Z]\d{4}\b',          # e.g. 22891A0678
            r'\b\d{2}[A-Z]{2}\d{4}\b',       # e.g. 21CS1001
            r'\b\d{2}[A-Z]{2}\d{6}\b',       # e.g. 19EC123456
            r'\b\d{4}BTECH\d{4}\b',           # e.g. 2019BTECH0001
            r'\b\d{12}\b'                    # e.g. 123456789012
        ]

        roll_number = None
        for pattern in patterns:
            rollno_match = re.search(pattern, text.upper())
            if rollno_match:
                roll_number = rollno_match.group(0).strip()
                break

        if roll_number:
            roll_status = f"✅ Roll Number Detected: {roll_number}"
        else:
            roll_status = "❌ Roll Number Not Detected"

        # Face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        face_status = f"✅ Face Detected ({len(faces)} face{'s' if len(faces) > 1 else ''})" if len(faces) > 0 else "❌ No Face Detected"

        # Print results
        print(college_status)
        print(name_status)
        print(roll_status)
        print(face_status)
