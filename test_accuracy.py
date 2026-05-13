import cv2
import numpy as np
import tensorflow as tf  # Make sure TensorFlow is imported

# Load your trained model (replace with your actual model loading code)
model = tf.keras.models.load_model('C:\\Users\\tanin\\OneDrive\\Desktop\\emerging_trends\\Sign Language\\asl_sign_language_modello.keras')

img_path = "C:\\Users\\tanin\\OneDrive\\Desktop\\emerging_trends\\Sign Language\\C.jpg" # Replace with the actual path
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is not None:
    img_resized = cv2.resize(img, (28, 28))
    img_normalized = img_resized / 255.0
    img_input = img_normalized.reshape(28, 28, 1)
    img_input = np.expand_dims(img_input, axis=0) 

    prediction = model.predict(img_input)
    predicted_label_index = np.argmax(prediction)

    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L',11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}  # Replace with your actual label map
    predicted_sign = label_map[predicted_label_index]
    print(f"The predicted sign is: {predicted_sign}")
else:
    print("Could not load image.")