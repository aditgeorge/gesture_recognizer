# from django.shortcuts import render
# from django.http import HttpResponse
import tensorflow
import tensorflow.python.autograph.pyct.anno
import tensorflow.python.keras.losses
from django.shortcuts import render
from polls.LSTM_MODEL import MY_LSTM
# from pathlib import Path
import cv2
# import time
import numpy as np
import tensorflow
import h5py
# from matplotlib import pyplot as plt
# import time
import mediapipe as mp
# from rest_framework.response import Response
from django.http.response import StreamingHttpResponse
# from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
# from scipy import stats
# from . import views
from django.conf import settings
# import tensorflow as tf
import os

# import urllib.request


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils


def index(request):
    return render(request, 'sili.html')


def sili(request):
    # return HttpResponse("SILI")
    return render(request, 'sili.html')


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)  # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)

                              )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


actions = np.array(['hello', 'thanks', 'iloveyou'])
print('hAIIIII')

# path1 = os.path.join(settings.MODELS, 'weights.h5')
# path2 = os.path.join(settings.MODELS, 'model.json')

# path1 = "C:\\Users\\sreya\\Downloads\\signlanguage\\SignLanguage\\polls\\models\\weights.h5"
# path2 = "C:\\Users\\sreya\\Downloads\\signlanguage\\SignLanguage\\polls\\models\\model.json"
label_map = {label: num for num, label in enumerate(actions)}
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
sequence = []
seq = []
sentence = []
predictions = []
# sentence = []
sent = []
threshold = 0.5
# json_file = open(path2, 'r')
# model_json = json_file.read()
# json_file.close()
# model = keras.models.model_from_json(model_json)
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.load_weights(path1)


# model = tensorflow.keras.models.Sequential()
# model.add(layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
# model.add(layers.LSTM(128, return_sequences=True, activation='relu'))  # return_sequences=True,
# model.add(layers.LSTM(64, return_sequences=False, activation='relu'))  # return_sequences=False,
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(actions.shape[0], activation='softmax'))
#
model = MY_LSTM()
print(model.summary())
#
# model.load_weights('action.h5')
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Hereeee")


# graph=tf.get_default_graph()
camera = cv2.VideoCapture(0)


def frame_create():
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while camera.isOpened():
            ret, frame = camera.read()
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            # print("Here")
            keypoints = extract_keypoints(results)
            global sequence
            sequence.append(keypoints)
            # print('lennn',len(sequence))
            sequence = sequence[-30:]

            if len(sequence) == 30:
                # print('SRHKLARG HAIIIIIIIIIIIIIIIII')
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                print(actions[np.argmax(res)])
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        global sentence
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    print(sentence)
                    sentence = sentence[-5:]

                image = prob_viz(res, actions, image, colors)
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            r, buf = cv2.imencode('.jpg', image)
            image = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n\r\n')


def video_gen(request):
    while True:
        return StreamingHttpResponse(frame_create(),
                                     content_type='multipart/x-mixed-replace; boundary=frame')
