from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import requests
import numpy as np

app = Flask(__name__)

# Global variables to control recognition and store current predictions
recognition_active = False
current_prediction = {} # Stores data for the last detected face

# --- Face Recognition Setup ---
PersonNames = ['Aditya']    # Must match order of labels in your trained model
# Using os.path.dirname(__file__) to ensure model path is relative to the script's location
face_recognizer_model_path = os.path.join(os.path.dirname(__file__), "face_recogonizer.yml")

# Create LBPH face recognizer and load trained data
my_model = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists(face_recognizer_model_path):
    try:
        my_model.read(face_recognizer_model_path)
        print("Loaded face recognizer model.")
    except cv2.error as e:
        print(f"Error loading face recognizer model: {e}")
        print("Please ensure 'face_recogonizer.yml' is a valid model file.")
        my_model = None # Set to None if loading fails to prevent further errors
else:
    print(f"Warning: Face recognizer model file '{face_recognizer_model_path}' not found! Please run train_age_gender.py first.")
    my_model = None

# Face Detector (Haar Cascade)
# CORRECTED TYPO: 'haascade' changed to 'haarcascade'
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_detector.empty(): # Check if the cascade classifier loaded correctly
    print("Error: Could not load Haar Cascade XML file. Face detection will not work.")
    print("Please ensure 'haarcascade_frontalface_default.xml' is available in your OpenCV data path.")
    face_detector = None # Set to None if it failed to load

# --- Age and Gender Detection Setup ---
# No longer using download_file function in app.py as we assume manual download
# You should ensure these files are in the same directory as app.py

# Local filenames for these models
# Using os.path.dirname(__file__) for robust path resolution
age_deploy_file = os.path.join(os.path.dirname(__file__), "age_deploy.prototxt")
age_net_file = os.path.join(os.path.dirname(__file__), "age_net.caffemodel")
gender_deploy_file = os.path.join(os.path.dirname(__file__), "gender_deploy.prototxt")
gender_net_file = os.path.join(os.path.dirname(__file__), "gender_net.caffemodel")

# Load Age and Gender models only if all necessary files are present
age_net = None
gender_net = None

# Explicitly check for file existence before attempting to load
age_files_exist = os.path.exists(age_deploy_file) and os.path.exists(age_net_file)
gender_files_exist = os.path.exists(gender_deploy_file) and os.path.exists(gender_net_file)

if age_files_exist and gender_files_exist:
    try:
        age_net = cv2.dnn.readNet(age_deploy_file, age_net_file)
        gender_net = cv2.dnn.readNet(gender_deploy_file, gender_net_file)
        print("Loaded Age and Gender detection models.")
    except cv2.error as e:
        print(f"Error loading DNN models: {e}")
        print(f"Ensure the model files are not corrupted and are in the correct directory: {os.path.dirname(__file__)}")
else:
    print("Warning: One or more Age/Gender model files are missing. Age/Gender prediction will not work.")
    if not age_files_exist:
        print(f"  Missing Age model files: {age_deploy_file} or {age_net_file}")
    if not gender_files_exist:
        print(f"  Missing Gender model files: {gender_deploy_file} or {gender_net_file}")


MODEL_MEAN_VALUES = (78.4263377603, 87.901088945, 114.5965258849)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(33-37)', '(38-43)', '(44-47)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# --- Video Streaming and Processing ---
def generate_frames():
    global current_prediction, recognition_active
    cap = cv2.VideoCapture(0) # 0 for default camera
    if not cap.isOpened():
        print("Error: Cannot access the camera. Please check if it's in use or connected.")
        # Return an empty frame or error image to Flask
        while True:
            # Create a black image with error text
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Error: Check connection", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret_err, buffer_err = cv2.imencode('.jpg', error_frame)
            frame_bytes_err = buffer_err.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes_err + b'\r\n')
            cv2.waitKey(1000) # Wait a bit before sending next error frame
        # No need for cap.release() as it failed to open

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame during runtime. Stream may have ended.")
            break # Exit loop if frame cannot be read

        # Reset the current prediction for this frame
        current_prediction = {}

        if recognition_active:
            # Check if face_detector loaded successfully before using it
            if face_detector is None:
                cv2.putText(frame, "Face Detector Error!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # Skip face detection and processing if detector is not loaded
                ret2, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue # Go to next frame

            # Convert to grayscale for face detection and LBPH recognition
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                # Get regions for recognition and age/gender detection
                face_roi_gray = gray[y:y+h, x:x+w]
                face_roi_color = frame[y:y+h, x:x+w]

                # --- Face Recognition (only if model loaded) ---
                name_text = "Unknown"
                if my_model:
                    label, confidence = my_model.predict(face_roi_gray)
                    threshold = 100   # Lower confidence values mean a more reliable match
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

                # --- Age & Gender Prediction (only if models loaded) ---
                age = "N/A"
                gender = "N/A"
                gender_conf = 0.0
                age_conf = 0.0

                if age_net and gender_net: # Only proceed if both age_net and gender_net loaded successfully
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
                    # Convert float32 to standard Python float for JSON serialization
                    "confidence": float(round(confidence, 2)) if my_model else "N/A",
                    "gender_confidence": float(round(gender_conf, 2)),
                    "age_confidence": float(round(age_conf, 2))
                }
                break   # Only process the first detected face for prediction

        else: # If recognition is not active
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
    status_msg = "Recognition started" if recognition_active else "Recognition stopped"
    print(status_msg)
    return jsonify({"status": "success", "recognition_active": recognition_active, "message": status_msg})

@app.route('/get_predictions')
def get_predictions():
    return jsonify(current_prediction)

if __name__ == '__main__':
    # Ensure Flask is run in a way that allows camera access
    # Use 0.0.0.0 to make it accessible from other devices on the network
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True) # threaded=True helps with camera stream stability