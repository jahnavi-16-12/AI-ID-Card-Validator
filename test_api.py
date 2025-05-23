import requests
import base64
import os

# ✅ Step 1: Set the correct path to your image
IMAGE_PATH = "input_test.jpg"  # Make sure this file exists

# ✅ Step 2: Check if the image exists
if not os.path.exists(IMAGE_PATH):
    print("❌ Image not found at path:", IMAGE_PATH)
    exit()

# ✅ Step 3: Read and encode image as base64
with open(IMAGE_PATH, "rb") as img_file:
    image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

# ✅ Step 4: Prepare the payload
payload = {
    "user_id": "stu_2290",
    "image_base64": image_base64
}

# ✅ Step 5: Send POST request to FastAPI endpoint
try:
    response = requests.post("http://127.0.0.1:8000/validate-id", json=payload)
    print("✅ Status Code:", response.status_code)
    print("✅ Response:", response.json())
except requests.exceptions.RequestException as e:
    print("❌ Request failed:", e)
