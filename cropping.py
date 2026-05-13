import cv2
import mediapipe as mp
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_root = os.path.join(BASE_DIR, "extra")
output_root = os.path.join(BASE_DIR, "clean_extra")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Create output root if it doesn't exist
os.makedirs(output_root, exist_ok=True)

# Process all subfolders (A–Y)
for label_folder in os.listdir(input_root):
    input_folder = os.path.join(input_root, label_folder)
    output_folder = os.path.join(output_root, label_folder)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        image = cv2.imread(input_path)
        if image is None:
            continue  # Skip unreadable images

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            h, w, _ = image.shape
            landmarks = results.multi_hand_landmarks[0].landmark
            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Crop without changing anything else
            cropped = image[y_min:y_max, x_min:x_max]
            cv2.imwrite(output_path, cropped)
        else:
            print(f"No hand detected in {input_path}")
