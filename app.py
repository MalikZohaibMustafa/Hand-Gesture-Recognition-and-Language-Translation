import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Updated labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'space'}

# Determine the expected feature length
expected_feature_length = 84

def process_image(image):
    frame = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    H, W, _ = frame.shape

    data_aux = []
    x_ = []
    y_ = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if len(data_aux) < expected_feature_length:
            data_aux.extend([0] * (expected_feature_length - len(data_aux)))

        if len(data_aux) == expected_feature_length:
            prediction = model.predict([np.asarray(data_aux)])
            # return labels_dict[int(prediction[0])]
            return (prediction[0])

    return None

@app.route('/classify', methods=['POST'])
def predict_single():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        predicted_character = process_image(file)
        if predicted_character:
            return jsonify({'predicted_character': predicted_character})
        return jsonify({'error': 'Hand is not properly aligned in frame or not present'})
    return jsonify({'error': 'Invalid file'})

@app.route('/classify_multiple', methods=['POST'])
def predict_multiple():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'})
    
    files = request.files.getlist('files')
    sentence = ""

    for file in files:
        if file and file.filename != '':
            predicted_character = process_image(file)
            if predicted_character:
                if predicted_character == 'space':
                    sentence += " "
                elif predicted_character == 'del':
                    sentence = sentence[:-1]
                else:
                    sentence += predicted_character
            else:
                return jsonify({'error': 'Hand is not properly aligned in frame or not present'})
    
    return jsonify({'sentence': sentence})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
