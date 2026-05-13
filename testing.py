import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model(r'C:\Users\tanin\OneDrive\Desktop\Emerging Trends\Sign Language\asl_sign_language_model.keras')

# Define your categories
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
 
img = cv2.imread("C:\\Users\\tanin\\OneDrive\\Desktop\\Emerging Trends\\Sign Language\\clean_extra\\A\\frame_70")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

img_resized = cv2.resize(img, (64, 64))
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY) 
img_normalized = img_gray / 255.0
img_reshaped = img_normalized.reshape(-1, 64, 64, 1)

plt.figure(figsize=(5,5))  
plt.imshow(img, cmap='gray') # grayscale mode
plt.title("Preprocessed Image (28x28)")
plt.axis('off')  
plt.show()


predictions = model.predict(img_reshaped)
predicted_label = np.argmax(predictions, axis=1)[0]


if 0 <= predicted_label < len(categories):
    print(f'Predicted gesture: {categories[predicted_label]}')
else:
    print('Unknown gesture')
