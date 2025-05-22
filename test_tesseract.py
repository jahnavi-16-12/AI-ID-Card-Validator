import pytesseract
from PIL import Image

# Print Tesseract version
print("Tesseract Version:", pytesseract.get_tesseract_version())

# Test with your input image
img = Image.open('input_test.jpg')
text = pytesseract.image_to_string(img)

print("\nComplete OCR Result:")
print("-" * 50)
print(text)
print("-" * 50)

# Test specific patterns
print("\nValidation Results:")
print("1. School/College name found:", "JANAPATH HIGHER SEC. SCHOOL" in text)
print("2. Contains 'Name' keyword:", "name" in text.lower())
print("3. Contains numbers (possible roll number):", any(c.isdigit() for c in text)) 