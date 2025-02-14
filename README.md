

# Hand-Sign-Detection-System-for-Specially-Abled

Below is an example README in Markdown that you can use for your GitHub repository. You can copy and paste this into your README.md file:

---

# Sign Language Detection with Python and Scikit-Learn

This repository contains a complete project on sign language detection using Python. The project focuses on recognizing three American Sign Language symbols (A, B, and L) by leveraging hand landmark detection. The approach uses:

- **OpenCV** for capturing and processing images.
- **MediaPipe** for extracting hand landmarks.
- **scikit-learn** for training a Random Forest classifier.

The focus is on extracting only the essential features (hand landmarks) rather than processing the full image, which results in a smaller, more robust, and efficient model.

## Project Structure

```
.
├── data/                  # Collected samples for each sign (A, B, L)
├── model/                 # Saved trained model (e.g., model.pkl)
├── scripts/               # Python scripts for data collection, training, and inference
└── README.md              # Project documentation
```

## Data Collection

The data collection phase captures webcam images of hand gestures:
- A script collects images as the user moves their hand closer to and away from the camera.
- Images for each sign (A, B, L) are stored in separate directories, with 100 samples per class.
- This diversity in samples helps build a robust dataset for the classifier.

## Landmark Extraction with MediaPipe

Instead of using full images, the project extracts hand landmarks to:
- Detect key points (approximately 20 per hand) that represent the positions of fingers and the hand.
- Significantly reduce the dimensionality of the input data by focusing solely on the (x, y) coordinates of the landmarks.
- Prepare the data for more efficient classification.

## Data Processing and Model Training

The extracted landmark data is then processed:
- The (x, y) coordinates for each landmark are arranged into arrays.
- Data is shuffled and split into training and test sets using a stratified train-test split.
- A **Random Forest classifier** is trained on this processed data.
- The model is evaluated on the test set, achieving high (often perfect) accuracy in this controlled setup.

### Example Training Code

```python
import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assume 'data' and 'labels' are prepared lists containing landmark arrays and their corresponding labels.
data = np.array(data)
labels = np.array(labels)

# Split the data into training and testing sets (20% test size)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(clf, f)
```

## Real-Time Detection

For live sign detection, the saved model is loaded and applied to webcam video:
- The system processes each frame to extract hand landmarks.
- The classifier predicts the sign based on the extracted landmark data.
- The predicted sign is overlayed on the video feed along with a bounding box around the hand.

### Example Inference Code

```python
import cv2
import mediapipe as mp
import pickle

# Load the trained model
with open('model/model.pkl', 'rb') as f:
    clf = pickle.load(f)

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract landmark positions
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y])
            # Predict the sign
            prediction = clf.predict([landmark_list])
            # Display the predicted sign on the frame
            cv2.putText(frame, f'Sign: {prediction[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Sign Language Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Requirements

- Python 3.6+
- OpenCV: `pip install opencv-python`
- MediaPipe: `pip install mediapipe`
- scikit-learn: `pip install scikit-learn`
- NumPy: `pip install numpy`

## How to Run

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/sign-language-detection.git
    cd sign-language-detection
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Data Collection:**
    - Run the data collection script in the `scripts/` directory to capture hand gesture images.

4. **Model Training:**
    - Execute the training script to extract landmarks, prepare the dataset, train the classifier, and save the model.

5. **Real-Time Inference:**
    - Run the inference script to test the model in real-time using your webcam.

## Conclusion

This project demonstrates a practical approach to sign language detection by focusing on the extraction of hand landmarks for feature reduction and using a Random Forest classifier for efficient prediction. The system is designed for real-time performance and offers a robust framework that can be expanded to include additional signs or more complex gesture recognition tasks.

Happy coding and feel free to contribute!

---

This format should work well for your GitHub repository, providing a clear explanation of your project along with example code and usage instructions.
