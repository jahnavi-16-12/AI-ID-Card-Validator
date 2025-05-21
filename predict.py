import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('model/image_model.keras')

# Class names
class_names = ['fake', 'genuine', 'suspicious']  # adjust if needed

# Load and preprocess image
image_path = "test_images/sample_id.jpg"
img = image.load_img(image_path, target_size=(224, 224))  # Match model input
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize if you did this during training
img_array = tf.expand_dims(img_array, 0)  # Shape becomes (1, 224, 224, 3)

# Predict
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
predicted_label = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

# Output
print(f"Predicted: {predicted_label} with {confidence:.2f}% confidence")
