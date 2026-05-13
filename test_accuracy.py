import cv2
import numpy as np
import tensorflow as tf
import os

# Setup Relative Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'asl_sign_language_model.keras')

# Load the model
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    print(f"Error: Model not found at {model_path}")
    exit()

# Define the Test Image path
img_path = os.path.join(BASE_DIR, "test.jpg") 

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is not None:
    # Preprocessing
    img_resized = cv2.resize(img, (28, 28))
    img_normalized = img_resized / 255.0
    img_input = img_normalized.reshape(28, 28, 1)
    img_input = np.expand_dims(img_input, axis=0) 

    # Prediction
    prediction = model.predict(img_input, verbose=0)
    predicted_label_index = np.argmax(prediction)

    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
                 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 
                 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}
    
    predicted_sign = label_map[predicted_label_index]
    confidence = np.max(prediction) * 100

    print(f"Prediction: {predicted_sign}")
    print(f"Confidence: {confidence:.2f}%")
else:
    print(f"Could not load image. Please place a 'test.jpg' in: {BASE_DIR}")
