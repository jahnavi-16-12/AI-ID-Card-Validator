import base64
image_base64 = base64.b64encode(open("input_test.jpg", "rb").read()).decode()
print(image_base64)
