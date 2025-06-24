from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from PIL import Image
import base64
from io import BytesIO
import mediapipe as mp

app = Flask(__name__)
CORS(app)

model = joblib.load('asl_svm3_model.pkl')

classes = sorted(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def preprocess_frame(frame, img_size=64):
    img_resized = cv2.resize(frame, (img_size, img_size))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges = np.uint8(np.abs(sobel_edges))
    return sobel_edges

def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

def extract_hand_roi(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            h, w, _ = image.shape
            hand_landmarks = results.multi_hand_landmarks[0]
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # إضافة هوامش بسيطة
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            return image[y_min:y_max, x_min:x_max]
    return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        img_bytes = base64.b64decode(data)
        image = Image.open(BytesIO(img_bytes)).convert('RGB')
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        roi = extract_hand_roi(frame)
        if roi is None:
            return jsonify({'error': 'Hand not detected'})

        processed = preprocess_frame(roi)
        features = extract_hog_features(processed)
        prediction = model.predict([features])[0]

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
