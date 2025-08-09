import cv2
import os
import requests
import numpy as np

# Model file locations - make sure these are in the same folder as this script
face_recognizer_model_path = os.path.join(os.path.dirname(__file__), "face_recognizer.yaml")
age_deploy_file = os.path.join(os.path.dirname(__file__), "age_deploy.prototxt")
age_net_file = os.path.join(os.path.dirname(__file__), "age_net.caffemodel")
gender_deploy_file = os.path.join(os.path.dirname(__file__), "gender_deploy.prototxt")
gender_net_file = os.path.join(os.path.dirname(__file__), "gender_net.caffemodel")

# Online links to download age and gender models if you don't have them
age_deploy_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
age_net_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_net.caffemodel"
gender_deploy_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
gender_net_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_net.caffemodel"

# Function to get model files – downloads if they're missing
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Getting {filename} from the internet...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Finished downloading {filename}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Couldn't download {filename}: {e}")
            print("Check your internet connection or the link.")
            return False
    else:
        print(f"{filename} is already here.")
        return True

# Load face detection tool (Haar Cascade)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_detector.empty():
    print("Error: Face detection tool couldn't load. It won't work.")
    exit()

# Dynamic PersonNames Loading
PersonNames = []
real_people_dir = os.path.join(os.path.dirname(__file__), "RealPeople")
if os.path.exists(real_people_dir):
    for f in os.listdir(real_people_dir):
        path = os.path.join(real_people_dir, f)
        if os.path.isdir(path):
            PersonNames.append(f)
    PersonNames.sort()
    print(f"Loaded {len(PersonNames)} recognized people: {PersonNames}")
else:
    print(f"Warning: 'RealPeople' directory not found at '{real_people_dir}'.")
    print("Face recognition will not work without trained data.")
    exit()

# Load person recognition model
my_model = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists(face_recognizer_model_path):
    try:
        my_model.read(face_recognizer_model_path)
        print("Person recognition model loaded.")
    except cv2.error as e:
        print(f"Error loading person recognition model: {e}")
        print("Make sure 'face_recognizer.yaml' is a proper model file from train_age_gender.py.")
        exit()
else:
    print(f"Error: Person recognition model file '{face_recognizer_model_path}' not found!")
    print("Please run train_age_gender.py first to create it.")
    exit()

# Load age and gender detection models
age_models_ready = download_file(age_deploy_url, age_deploy_file) and download_file(age_net_url, age_net_file)
gender_models_ready = download_file(gender_deploy_url, gender_deploy_file) and download_file(gender_net_url, gender_net_file)

age_net = None
gender_net = None

if age_models_ready and gender_models_ready:
    try:
        age_net = cv2.dnn.readNet(age_deploy_file, age_net_file)
        gender_net = cv2.dnn.readNet(gender_deploy_file, gender_net_file)
        print("Age and gender models loaded.")
    except cv2.error as e:
        print(f"Error loading age/gender models: {e}")
        print("Check if the model files are correct and in the right place.")
else:
    print("Warning: Age/Gender prediction won't work because some model files are missing or failed to download.")

# Data for age and gender prediction
MODEL_MEAN_VALUES = (78.4263377603, 87.901088945, 114.5965258849)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(33-37)', '(38-43)', '(44-47)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# UI display settings
COLOR_RECOGNIZED = (10, 255, 10)
COLOR_UNKNOWN = (0, 0, 255)
COLOR_TEXT_BG = (50, 50, 50)
COLOR_TEXT = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
PADDING = 5
LINE_HEIGHT_OFFSET = 30

# Start camera feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Can't open the camera. Is it being used or unplugged?")
    exit()

print("\nStarting face recognition. Press 'q' to stop.")

window_name = 'Face Recognition'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Couldn't get camera frame. Exiting.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]

        name_text = "Unknown"
        confidence_display = "N/A"
        
        if my_model:
            label, confidence = my_model.predict(face_roi_gray)
            
            if confidence < 100:
                name_text = PersonNames[label]
                percentage_confidence = max(0, min(100, 100 - (confidence * 0.5)))
                confidence_display = f"Conf: {percentage_confidence:.2f}%"
            else:
                name_text = "Unknown"
                confidence_display = "N/A"

        box_color = COLOR_RECOGNIZED if name_text != "Unknown" else COLOR_UNKNOWN
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

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

        current_y_offset = y + h + 10
        info_lines = [
            f"Name: {name_text}",
            confidence_display,
            f"Gender: {gender}",
            f"Age: {age}"
        ]

        for line in info_lines:
            if line != "N/A":
                (text_width, text_height), baseline = cv2.getTextSize(line, FONT, FONT_SCALE, FONT_THICKNESS)
                cv2.rectangle(frame, (x, current_y_offset), (x + text_width + 2 * PADDING, current_y_offset + text_height + 2 * PADDING), COLOR_TEXT_BG, cv2.FILLED)
                cv2.putText(frame, line, (x + PADDING, current_y_offset + text_height + PADDING), FONT, FONT_SCALE, COLOR_TEXT, FONT_THICKNESS)
                current_y_offset += (text_height + 2 * PADDING) + PADDING

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()