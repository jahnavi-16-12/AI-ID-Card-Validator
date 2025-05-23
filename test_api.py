import requests
import base64
import os

# Path to the image you want to test
IMAGE_PATH = "input_test.jpg"  # Change this to your image file path

# Check if image file exists
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Image not found at path: {IMAGE_PATH}")
    exit()

# Read and encode image as base64 string
with open(IMAGE_PATH, "rb") as img_file:
    image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

# Prepare the JSON payload to send
payload = {
    "user_id": "stu_2290",
    "image_base64": image_base64
}

# URL of your running FastAPI server
API_URL = "http://127.0.0.1:8000/validate-id"

try:
    response = requests.post(API_URL, json=payload)
    print("✅ Status Code:", response.status_code)
    print("✅ Response JSON:", response.json())
except requests.exceptions.RequestException as e:
    print("❌ Request failed:", e)
