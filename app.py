import os
import cv2
import requests
import numpy as np
import threading
import time
import json

from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
from queue import Queue

app = Flask(__name__, template_folder='templates')
CORS(app)

# --- Global variables to control recognition and store current predictions ---
recognition_active = False
# Use a Queue for thread-safe prediction updates
prediction_queue = Queue()

# --- Model Paths ---
face_recognizer_model_path = os.path.join(os.path.dirname(__file__), "face_recognizer.yaml")
age_deploy_file = os.path.join(os.path.dirname(__file__), "age_deploy.prototxt")
age_net_file = os.path.join(os.path.dirname(__file__), "age_net.caffemodel")
gender_deploy_file = os.path.join(os.path.dirname(__file__), "gender_deploy.prototxt")
gender_net_file = os.path.join(os.path.dirname(__file__), "gender_net.caffemodel")

# --- Age and Gender Detection Model URLs (for initial download) ---
age_deploy_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
age_net_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_net.caffemodel"
gender_deploy_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
gender_net_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_net.caffemodel"

# --- Helper Function to Download Models ---
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Downloaded {filename} successfully.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename} from {url}: {e}")
            print("Please check your internet connection or the URL.")
            return False
    else:
        print(f"{filename} already exists.")
    return True

# --- Global Model Variables ---
my_model = None
face_detector = None
age_net = None
gender_net = None

# --- Dynamic PersonNames Loading ---
PersonNames = []
real_people_dir = os.path.join(os.path.dirname(__file__), "RealPeople")
def load_person_names():
    global PersonNames
    PersonNames = []
    if os.path.exists(real_people_dir):
        for f in os.listdir(real_people_dir):
            path = os.path.join(real_people_dir, f)
            if os.path.isdir(path):
                PersonNames.append(f)
        PersonNames.sort()
        print(f"Loaded {len(PersonNames)} recognized people from 'RealPeople' directory: {PersonNames}")
    else:
        print(f"Warning: 'RealPeople' directory not found at '{real_people_dir}'.")
        print("Face recognition will not work without trained data.")
load_person_names()

# --- Function to Load All Models ---
def load_all_models():
    global my_model, face_detector, age_net, gender_net
    load_person_names()

    if os.path.exists(face_recognizer_model_path):
        try:
            my_model = cv2.face.LBPHFaceRecognizer_create()
            my_model.read(face_recognizer_model_path)
            print("Loaded face recognizer model (face_recognizer.yaml).")
        except cv2.error as e:
            print(f"Error loading face recognizer model: {e}")
            print("Please ensure 'face_recognizer.yaml' is a valid model file from train_age_gender.py.")
            my_model = None
    else:
        print(f"Warning: Face recognizer model file '{face_recognizer_model_path}' not found!")
        print("Please run train_age_gender.py first to train the model.")
        my_model = None

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_detector.empty():
        print("Error: Could not load Haar Cascade XML file. Face detection will not work.")
        face_detector = None

    age_models_ready = download_file(age_deploy_url, age_deploy_file) and download_file(age_net_url, age_net_file)
    gender_models_ready = download_file(gender_deploy_url, gender_deploy_file) and download_file(gender_net_url, gender_net_file)

    if age_models_ready and gender_models_ready:
        try:
            age_net = cv2.dnn.readNet(age_deploy_file, age_net_file)
            gender_net = cv2.dnn.readNet(gender_deploy_file, gender_net_file)
            print("Loaded Age and Gender detection models.")
        except cv2.error as e:
            print(f"Error loading DNN models: {e}")
            print("Ensure the model files are not corrupted and are in the correct directory.")
            age_net = None
            gender_net = None
    else:
        print("Warning: One or more Age/Gender model files are missing or failed to download. Age/Gender prediction will not work.")

# Model specific constants
MODEL_MEAN_VALUES = (78.4263377603, 87.901088945, 114.5965258849)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(33-37)', '(38-43)', '(44-47)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# UI display settings (used for drawing on frames)
COLOR_RECOGNIZED = (10, 255, 10)
COLOR_UNKNOWN = (0, 0, 255)
COLOR_TEXT_BG = (50, 50, 50)
COLOR_TEXT = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
PADDING = 5
LINE_HEIGHT_OFFSET = 30

# --- Video Streaming and Processing Function (MAIN LOGIC) ---
def generate_frames():
    global recognition_active

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access the camera. Please check if it's in use or connected.")
        while True:
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "CAMERA ERROR", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            ret_err, buffer_err = cv2.imencode('.jpg', error_frame)
            frame_bytes_err = buffer_err.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes_err + b'\r\n')
            time.sleep(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame during runtime.")
                break

            all_faces_data_for_frame = []
            frame_to_display = frame.copy()

            if recognition_active:
                if face_detector is None:
                    cv2.putText(frame_to_display, "Face Detector Error!", (20, 60), FONT, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

                    for (x, y, w, h) in faces:
                        face_roi_gray = gray[y:y+h, x:x+w]
                        face_roi_color = frame[y:y+h, x:x+w]

                        name_text = "Unknown"
                        confidence_display = "N/A"
                        confidence_val = 1000

                        if my_model and len(PersonNames) > 0:
                            label, conf_val = my_model.predict(face_roi_gray)
                            threshold = 100
                            confidence_val = conf_val

                            if conf_val < threshold:
                                name_text = PersonNames[label]
                                percentage_confidence = max(0, min(100, 100 - (conf_val * 0.5)))
                                confidence_display = f"{percentage_confidence:.2f}%"
                            else:
                                name_text = "Unknown"
                                confidence_display = "N/A"
                        elif my_model:
                            name_text = "Unknown (No trained people)"
                            confidence_display = "N/A"

                        box_color = COLOR_RECOGNIZED if name_text != "Unknown" and "trained people" not in name_text else COLOR_UNKNOWN
                        cv2.rectangle(frame_to_display, (x, y), (x+w, y+h), box_color, 2)

                        age = "N/A"
                        gender = "N/A"

                        if age_net and gender_net:
                            blob = cv2.dnn.blobFromImage(face_roi_color, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                            gender_net.setInput(blob)
                            gender_preds = gender_net.forward()
                            gender = gender_list[gender_preds[0].argmax()]

                            age_net.setInput(blob)
                            age_preds = age_net.forward()
                            age = age_list[age_preds[0].argmax()]
                        else:
                            cv2.putText(frame_to_display, "Age/Gender Models Not Loaded!", (x, y - 50), FONT, 0.6, (0, 0, 255), 1)

                        current_y_offset = y + h + 10
                        info_lines = [
                            f"Name: {name_text}",
                            f"Confidence: {confidence_display}",
                            f"Gender: {gender}",
                            f"Age: {age}"
                        ]
                        for line in info_lines:
                            (text_width, text_height), baseline = cv2.getTextSize(line, FONT, FONT_SCALE, FONT_THICKNESS)
                            cv2.rectangle(frame_to_display, (x, current_y_offset), (x + text_width + 2 * PADDING, current_y_offset + text_height + 2 * PADDING), COLOR_TEXT_BG, cv2.FILLED)
                            cv2.putText(frame_to_display, line, (x + PADDING, current_y_offset + text_height + PADDING), FONT, FONT_SCALE, COLOR_TEXT, FONT_THICKNESS)
                            current_y_offset += (text_height + 2 * PADDING) + PADDING

                        all_faces_data_for_frame.append({
                            "name": name_text,
                            "age": age,
                            "gender": gender,
                            "confidence": confidence_display,
                            "confidence_val": confidence_val # Raw confidence for better sorting/logic if needed
                        })
            else:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                alpha = 0.4
                frame_to_display = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                text_stopped = "Recognition Stopped"
                (text_width, text_height), baseline = cv2.getTextSize(text_stopped, FONT, FONT_SCALE * 1.5, FONT_THICKNESS * 2)
                text_x = int((frame.shape[1] - text_width) / 2)
                text_y = int((frame.shape[0] + text_height) / 2)
                cv2.putText(frame_to_display, text_stopped, (text_x, text_y), FONT, FONT_SCALE * 1.5, (0, 255, 255), FONT_THICKNESS * 2)
                all_faces_data_for_frame = [] # Clear predictions when stopped

            # Push the predictions to the queue
            prediction_queue.put({"faces": all_faces_data_for_frame})

            ret2, buffer = cv2.imencode('.jpg', frame_to_display)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_recognition', methods=['POST'])
def toggle_recognition():
    global recognition_active
    data = request.get_json()
    new_status = data.get("active", False)

    if new_status and not recognition_active:
        print("Starting recognition...")
        load_all_models() # Reload models to ensure they're ready and up-to-date
    
    recognition_active = new_status
    status_msg = "Recognition started" if recognition_active else "Recognition stopped"
    print(status_msg)

    # Immediately push a status update
    prediction_queue.put({"faces": [], "status": status_msg})

    return jsonify({"status": "success", "recognition_active": recognition_active, "message": status_msg})

@app.route('/stream_predictions')
def stream_predictions():
    def event_stream():
        while True:
            # Wait for a new prediction from the queue
            predictions = prediction_queue.get()
            yield f"data: {json.dumps(predictions)}\n\n"
            prediction_queue.task_done()
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == '__main__':
    load_all_models()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)