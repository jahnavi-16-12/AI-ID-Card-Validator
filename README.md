# AI ID Card Validator

AI ID Card Validator is an advanced web application that uses artificial intelligence, computer vision, and OCR to verify the authenticity of identity documents. It provides enterprise-grade security, fast processing, and high accuracy for document validation.

## Features
- **AI-Powered ID Card Validation**: Uses ONNX deep learning models to classify ID cards as genuine, fake, or suspicious.
- **OCR Text Extraction**: Extracts and validates text fields from ID cards using intelligent OCR.
- **Template Matching**: Checks if the ID card matches known templates.
- **Multi-layer Security**: Detects tampering, forgery, and manipulation.
- **REST API**: Exposes endpoints for validation and health checks.
- **Modern Frontend**: Responsive UI for uploading and validating ID cards.
- **Enterprise-Grade Security**: CORS, secure file handling, and robust error logging.

## Project Structure
```
AI ID Card Validator/
├── main.py                # FastAPI backend server
├── schemas.py             # Pydantic models for API requests/responses
├── ocr_validator.py       # OCR and field validation logic
├── template_matcher.py    # Template matching logic
├── decision.py            # Decision logic for final label/status
├── image_classifier.py    # (Optional) Image classification helpers
├── config.json            # Configuration (class names, thresholds, etc.)
├── approved_colleges.json # List of approved colleges for validation
├── requirements.txt       # Python dependencies
├── static/                # Static files (HTML, CSS, JS, images)
│   ├── index.html         # Main landing page
│   ├── validator.html     # Validator UI page
│   └── styles.css         # Stylesheet
├── templates/             # (Optional) Jinja2 templates
├── model/
│   └── image_model.onnx   # ONNX model for image classification
├── generated_ids/         # Output folders for generated IDs
├── photos/                # Sample/test images
├── tests/                 # Test scripts
└── ...
```

## Getting Started

### Prerequisites
- Python 3.11+
- pip

### Installation
1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd "AI ID Card Validator"
   ```
2. **Create a virtual environment (recommended):**
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On Linux/Mac
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

### Running the Application
1. **Start the FastAPI server:**
   ```sh
   uvicorn main:app --reload
   ```
2. **Open your browser:**
   - Main page: [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - Validator: [http://127.0.0.1:8000/validator](http://127.0.0.1:8000/validator)

### API Endpoints
- `GET /` : Main landing page (HTML)
- `GET /validator` : Validator UI page (HTML)
- `POST /validate-id` : Validate an ID card (expects base64 image, returns validation result)
- `GET /health` : Health check
- `GET /version` : API version


### How the Validation Pipeline Works
The validation process consists of three main steps:

1. **Image Classification**
   - The uploaded ID card image is first analyzed by an ONNX-based deep learning model.
   - The model classifies the image as `genuine`, `fake`, or `suspicious` based on visual features.

2. **OCR Validation**
   - Optical Character Recognition (OCR) is performed on the ID card image.
   - Extracted text fields (such as name, roll number, and college) are validated for correctness and completeness.
   - The college name is checked against an approved list.

3. **Template Matching**
   - The system compares the ID card layout and design with standard templates using the template matcher.
   - This step helps detect tampering, forgery, or non-standard document formats.

The results from all three steps are combined to make a final decision, which is returned to the user along with confidence scores.

### Usage
- Drag and drop an ID card image on the web UI or use the `/validate-id` API endpoint with a base64-encoded image.
- The system will classify the ID, extract and validate fields, check templates, and return a decision with confidence scores.

## Configuration
- **`config.json`**: Set class names, thresholds, and OCR requirements.
- **`approved_colleges.json`**: List of valid colleges for field validation.
- **`model/image_model.onnx`**: Replace with your own ONNX model if needed.

## Customization
- Update `static/styles.css` for UI changes.
- Add new templates or modify validation logic in `template_matcher.py` and `decision.py`.

## Testing
- Place test images in the `photos/` directory.
- Use the `tests/` folder for test scripts.

## License
This project is for educational and demonstration purposes. For production use, review and update security, privacy, and compliance requirements.

---
**© AI ID Card Validator. Powered by Advanced AI, Image Classification, and OCR Technology. Developed by Jahnavi Goud.**
