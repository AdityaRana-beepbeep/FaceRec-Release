from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import requests
import numpy as np

app = Flask(__name__)

# Global variables
recognition_active = False
current_prediction = {}

# --- Face Recognition Setup ---
PersonNames = ['Aditya']  # Must match order of labels in your trained model
face_recognizer_model_path = "face_recogonizer.yml"

# Create LBPH face recognizer and load trained data
my_model = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists(face_recognizer_model_path):
    my_model.read(face_recognizer_model_path)
    print("Loaded face recognizer model.")
else:
    print("Warning: Face recognizer model file not found!")

# Face Detector (Haar Cascade)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Age and Gender Detection Setup ---
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists.")

# URLs for the DNN models (from OpenCV's GitHub)
age_deploy_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/age_deploy.prototxt"
age_net_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/age_net.caffemodel"
gender_deploy_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/gender_deploy.prototxt"
gender_net_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/gender_net.caffemodel"

# Local filenames for these models
age_deploy_file = "age_deploy.prototxt"
age_net_file = "age_net.caffemodel"
gender_deploy_file = "gender_deploy.prototxt"
gender_net_file = "gender_net.caffemodel"

# Download models if needed
download_file(age_deploy_url, age_deploy_file)
download_file(age_net_url, age_net_file)
download_file(gender_deploy_url, gender_deploy_file)
download_file(gender_net_url, gender_net_file)

# Load Age and Gender models
age_net = cv2.dnn.readNet(age_deploy_file, age_net_file)
gender_net = cv2.dnn.readNet(gender_deploy_file, gender_net_file)

MODEL_MEAN_VALUES = (78.4263377603, 87.901088945, 114.5965258849)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(33-37)', '(38-43)', '(44-47)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# --- Video Streaming and Processing ---
def generate_frames():
    global current_prediction, recognition_active
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reset the current prediction for this frame
        current_prediction = {}

        if recognition_active:
            # Convert to grayscale for face detection and LBPH recognition
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                # Get regions for recognition and age/gender detection
                face_roi_gray = gray[y:y+h, x:x+w]
                face_roi_color = frame[y:y+h, x:x+w]

                # --- Face Recognition ---
                label, confidence = my_model.predict(face_roi_gray)
                threshold = 100  # Lower confidence values mean a more reliable match
                if confidence < threshold:
                    name_text = PersonNames[label]
                else:
                    name_text = "Unknown"

                # Draw rectangle and name
                if name_text != "Unknown":
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (10, 255, 10), 2)
                    cv2.putText(frame, name_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 255, 10), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, name_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # --- Age & Gender Prediction ---
                blob = cv2.dnn.blobFromImage(face_roi_color, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                
                # Gender prediction
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                gender_conf = gender_preds[0][gender_preds[0].argmax()]

                # Age prediction
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                age_conf = age_preds[0][age_preds[0].argmax()]

                info_text = f"{gender}, {age}"
                cv2.putText(frame, info_text, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 20), 2)
                
                # Update the global prediction with the first face's data found
                current_prediction = {
                    "name": name_text,
                    "age": age,
                    "gender": gender,
                    "confidence": round(confidence, 2),
                    "gender_confidence": round(gender_conf, 2),
                    "age_confidence": round(age_conf, 2)
                }
                break  # Only process the first detected face for prediction

        else:
            cv2.putText(frame, "Recognition Stopped", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Encode frame as JPEG
        ret2, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

# --- Flask Routes ---
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
    recognition_active = data.get("active", False)
    return jsonify({"status": "success", "recognition_active": recognition_active})

@app.route('/get_predictions')
def get_predictions():
    return jsonify(current_prediction)

if __name__ == '__main__':
    app.run(debug=True)
