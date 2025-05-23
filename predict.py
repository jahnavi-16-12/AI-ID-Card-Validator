import tensorflow as tf
import tf2onnx
import os

print("Current working directory:", os.getcwd())

seq_model = tf.keras.models.load_model('model/image_model.keras')

inputs = tf.keras.Input(shape=(224, 224, 3), name="input")
outputs = seq_model(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="converted_model")

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

print(type(onnx_model))
print(len(onnx_model.SerializeToString()))

output_path = "D:/AI-ID-Card-Validator/image_model.onnx"
with open(output_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ Conversion complete. Model saved as '{output_path}'")
