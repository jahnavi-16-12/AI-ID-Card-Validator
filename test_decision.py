import json
from decision import decide_label

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Inputs from your pipeline
validation_score = 0.88  # from image_classifier.py
ocr_fields_found = ["name", "roll_number", "college"]  # from ocr_validator.py
template_match_score = 0.23  # from template_matcher.py

# Make decision
label, decision, reason = decide_label(validation_score, ocr_fields_found, template_match_score, config)

print(f"Label: {label}")
print(f"Decision: {decision}")
print(f"Reason: {reason}")
