import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque, Counter
import pyttsx3

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get the directory where camera.py is located
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'asl_sign_language_model.keras')

# Load the model using the relative path
model = tf.keras.models.load_model(model_path)

categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

cap = cv2.VideoCapture(0)

# Text-to-Speech engine setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize buffer and tracking variables
prediction_queue = deque(maxlen=15)
last_stable_prediction = ""
word_buffer = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            #mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the bounding box of the hand
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            # Scale the bounding box coordinates to pixel values
            h, w, _ = frame.shape
            x_min = int(x_min * w)
            x_max = int(x_max * w)
            y_min = int(y_min * h)
            y_max = int(y_max * h)

            # Crop the hand region from the frame
            hand_region = frame[y_min:y_max, x_min:x_max]

            if hand_region.size != 0:
                hand_resized = cv2.resize(hand_region, (28, 28), interpolation=cv2.INTER_AREA)
                hand_gray = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2GRAY)

                # Normalize and reshape the image for prediction
                hand_normalized = hand_gray / 255.0
                hand_reshaped = hand_normalized.reshape(-1, 28, 28, 1)

                # Predict gesture
                predictions = model.predict(hand_reshaped, verbose=0)
                confidence = np.max(predictions)
                predicted_label = np.argmax(predictions, axis=1)[0]

                if confidence > 0.8:
                    predicted_gesture = categories[predicted_label]
                else:
                    predicted_gesture = "Unknown"

                # Add the prediction to the queue
                prediction_queue.append(predicted_gesture)

                most_common = Counter(prediction_queue).most_common(1)[0]
                if most_common[1] > 10:
                    stable_prediction = most_common[0]
                else:
                    stable_prediction = "..."

                if stable_prediction not in ["...", "Unknown"] and stable_prediction != last_stable_prediction:
                    word_buffer += stable_prediction
                    last_stable_prediction = stable_prediction

                cv2.putText(frame, f'Predicted: {predicted_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


    cv2.putText(frame, f'Buffer: {word_buffer}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


    cv2.imshow("Webcam Feed", frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  
        if word_buffer:
            engine.say(word_buffer)
            engine.runAndWait()
            word_buffer = ""  
    elif key == ord('r'): 
        word_buffer = ""


cap.release()
cv2.destroyAllWindows()
