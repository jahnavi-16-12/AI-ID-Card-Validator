# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
# from tensorflow.keras.preprocessing import image_dataset_from_directory
# from tensorflow.keras.preprocessing import image
# # Add this near top with imports
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal_and_vertical"),
#     tf.keras.layers.RandomRotation(0.2),
#     tf.keras.layers.RandomZoom(0.1),
# ])

# # Modify build_model()
# def build_model():
#     model = Sequential([
#         Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
#         data_augmentation,
#         Conv2D(32, (3,3), activation='relu'),
#         MaxPooling2D(2,2),
#         Conv2D(64, (3,3), activation='relu'),
#         MaxPooling2D(2,2),
#         Conv2D(128, (3,3), activation='relu'),
#         MaxPooling2D(2,2),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(3, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model


# # Constants
# IMAGE_SIZE = (224, 224)
# BATCH_SIZE = 32
# MODEL_PATH = "model/image_model.keras"
# DATA_DIR = "generated_ids/"

# # Load dataset with normalization and prefetching
# def load_data(data_dir=DATA_DIR):
#     train_ds = image_dataset_from_directory(
#         data_dir,
#         validation_split=0.2,
#         subset="training",
#         seed=123,
#         image_size=IMAGE_SIZE,
#         batch_size=BATCH_SIZE,
#         label_mode="categorical"
#     )
#     val_ds = image_dataset_from_directory(
#         data_dir,
#         validation_split=0.2,
#         subset="validation",
#         seed=123,
#         image_size=IMAGE_SIZE,
#         batch_size=BATCH_SIZE,
#         label_mode="categorical"
#     )
    
#     # Normalize pixel values to [0, 1]
#     normalization_layer = tf.keras.layers.Rescaling(1./255)
#     train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
#     val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    
#     return train_ds, val_ds

# # Build CNN model with Input layer
# def build_model():
#     model = Sequential([
#         Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
#         Conv2D(32, (3,3), activation='relu'),
#         MaxPooling2D(2,2),
#         Conv2D(64, (3,3), activation='relu'),
#         MaxPooling2D(2,2),
#         Conv2D(128, (3,3), activation='relu'),
#         MaxPooling2D(2,2),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(3, activation='softmax')  # 3 classes: genuine, fake, suspicious
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # Train model and save
# def train_and_save_model():
#     train_ds, val_ds = load_data()
#     model = build_model()
#     model.fit(train_ds, validation_data=val_ds, epochs=20)
#     os.makedirs("model", exist_ok=True)
#     model.save(MODEL_PATH)
#     print(f"Model saved to {MODEL_PATH}")

# # Load saved model
# def load_trained_model():
#     return load_model(MODEL_PATH)

# # Predict class of image with label mapping from dataset class_names
# def predict_image_class(img_path, model, class_names):
#     img = image.load_img(img_path, target_size=IMAGE_SIZE)
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     predictions = model.predict(img_array)
#     predicted_index = np.argmax(predictions[0])
#     confidence = float(np.max(predictions[0]))
#     label = class_names[predicted_index]
#     return label, confidence

# # Main execution block
# if __name__ == "__main__":
#     # Train the model (uncomment when training)
#      train_and_save_model()
    
#     # Load model and predict (comment out training above when running this)
#     #model = load_trained_model()
    
#     # Load class names dynamically
#     #dataset = image_dataset_from_directory(DATA_DIR, batch_size=1)
#     #class_names = dataset.class_names
    
#     # Example test image path
#     #test_image_path = "test_images/sample_id.jpg"  # change to your test image
    
#     #label, confidence = predict_image_class(test_image_path, model, class_names)
#     #print(f"Predicted: {label} with confidence {confidence:.2f}")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing import image_dataset_from_directory, image

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "model/image_model.keras"
DATA_DIR = "generated_ids/"

# Data augmentation (optional)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
])

# Build CNN model
def build_model():
    model = Sequential([
        Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        data_augmentation,
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # Classes: genuine, fake, suspicious
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load dataset with normalization and prefetching
def load_data(data_dir=DATA_DIR):
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    
    # Normalize pixel values
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

# Train and save model
def train_and_save_model():
    train_ds, val_ds = load_data()
    model = build_model()
    model.fit(train_ds, validation_data=val_ds, epochs=20)
    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Load saved model
def load_trained_model():
    return load_model(MODEL_PATH)

# ✅ Predict class of image from PIL object
def predict_image_class(img_pil, model, class_names):
    img = img_pil.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    label = class_names[predicted_index]
    return label, confidence

# Optional training/test entry point
if __name__ == "__main__":
    # Uncomment to train
    # train_and_save_model()

    # Example test call
    # model = load_trained_model()
    # from PIL import Image
    # img = Image.open("test_images/sample_id.jpg")
    # label, confidence = predict_image_class(img, model, ['fake', 'genuine', 'suspicious'])
    # print(f"Predicted: {label} with {confidence:.2f} confidence")
    pass
