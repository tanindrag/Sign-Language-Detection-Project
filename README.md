# Real-Time Sign Language to Speech Translator

### Project Overview
This application uses **Computer Vision** and **Deep Learning** to recognize American Sign Language (ASL) finger-spelling (Letters A-Y, excluding J and Z) in real-time. 
The system captures live video through a webcam, predicts the gesture using a trained Convolutional Neural Network (CNN), and converts the result into audible speech.

This project showcases a complete end-to-end pipeline: from data preprocessing and automation to model deployment and real-time inference.

---

### Key Features
* **Real-Time Inference:** Fast and responsive gesture classification via webcam.
* **Audio Synthesis:** Integrated `pyttsx3` to translate recognized signs into spoken words.
* **Hand Detection:** Utilizes MediaPipe for accurate hand tracking and region-of-interest cropping.
* **Automation Tools:** Custom Python utilities for batch-organizing and preprocessing image data.

---

### Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV, MediaPipe
* **Machine Learning:** TensorFlow, Keras, NumPy
* **Audio Engine:** Pyttsx3

---

### Repository Structure
* **main.py** - The primary entry point for the real-time application and webcam interface.
* **model_training_workflow.ipynb** - Technical documentation of the training process, including CNN architecture, data visualization, and final accuracy metrics.
* **asl_sign_language_model.keras** - The final pre-trained Deep Learning model ready for deployment.
* **preprocess.py** - Utility for hand-detection and automated image cropping to maintain data consistency.
* **organize_data.py** - Automation script for sorting raw image frames into labeled directories.
* **requirements.txt** - List of Python dependencies for environment setup.

---

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/tanindrag/Sign-Language-Detection-Project.git](https://github.com/tanindrag/Sign-Language-Detection-Project.git)
   cd Sign-Language-Detection-Project
   
2. **Install dependencies:**
     pip install -r requirements.txt

### Usage
Run the live translator:
   ```bash
   python main.py
   ```

### How it works:

1. The application opens a webcam feed window.
2. Position your hand within the frame.
3. The system predicts the ASL letter.
4. Press 'Enter' to have the system announce the predicted letter via your speakers (Text-to-Speech).
5. Press q to exit the application.

Note: The raw training dataset is excluded from this repository to maintain a lightweight project size and data privacy. However, the pre-trained model and training workflow are included for full transparency.

### Future Roadmap
1. Coordinate-Based Training: Shifting from raw pixels to MediaPipe landmark coordinates to improve accuracy in varied lighting.
2. Sentence Construction: Implementing NLP to group recognized letters into meaningful words and sentences.
3. Motion Recognition: Adding LSTM layers to recognize dynamic signs like 'J' and 'Z'.

📜 License
Distributed under the MIT License. See LICENSE file for details.
