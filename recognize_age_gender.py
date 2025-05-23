import cv2
import os
import requests

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array



PersonNames = ['Aditya'] #names
cap = cv2.VideoCapture(0)

my_model = cv2.face.LBPHFaceRecognizer_create()
my_model.read("face_recogonizer.yml")

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Age and Gender Detection Model URLs
age_deploy_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/age_deploy.prototxt"
age_net_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/age_net.caffemodel"
gender_deploy_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/gender_deploy.prototxt"
gender_net_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/gender_net.caffemodel"

age_deploy_file = "age_deploy.prototxt"
age_net_file = "age_net.caffemodel"
gender_deploy_file = "gender_deploy.prototxt"
gender_net_file = "gender_net.caffemodel"

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

# Download the model files if they don't exist
download_file(age_deploy_url, age_deploy_file)
download_file(age_net_url, age_net_file)
download_file(gender_deploy_url, gender_deploy_file)
download_file(gender_net_url, gender_net_file)

# --- Age and Gender Detection Models ---
age_net = cv2.dnn.readNet(age_deploy_file, age_net_file)
gender_net = cv2.dnn.readNet(gender_deploy_file, gender_net_file)

MODEL_MEAN_VALUES = (78.4263377603, 87.901088945, 114.5965258849)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(33 - 37)', '(38-43)', '(44-47)' '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Create a named window with normal flag to allow resizing and fullscreen
cv2.namedWindow('face', cv2.WINDOW_NORMAL)
# Set the window to fullscreen
cv2.setWindowProperty('face', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    x, frame = cap.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_detector.detectMultiScale(gray_img, 1.4, 3)

    for fx, fy, fw, fh in face:
        face_roi_color = frame[fy:fy + fh, fx:fx + fw] # Color ROI for age/gender

        label, confidence = my_model.predict(gray_img[fy:fy + fh, fx:fx + fw])
        name_text = PersonNames[label]
        confidence_text = f"Confidence: {round(confidence, 2)}"

        confidence_threshold = 100  # You can tweak this value

        if confidence < confidence_threshold:
        # Recognized
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (10, 255, 10), 3)
            cv2.putText(frame, name_text, (fx, fy - 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 20), 2)
            cv2.putText(frame, confidence_text, (fx, fy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 20), 2)
        else:
        # Unrecognized
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 3)
            cv2.putText(frame, "Unknown", (fx, fy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # --- Age and Gender Prediction ---
        blob = cv2.dnn.blobFromImage(face_roi_color, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender Prediction
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        gender_confidence = gender_preds[0][gender_preds[0].argmax()]

        # Age Prediction
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        age_confidence = age_preds[0][age_preds[0].argmax()]

        age_gender_text = f"{gender}, {age}"
        cv2.putText(frame, age_gender_text, (fx, fy + fh + 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 20), 2)
        confidence_age_gender_text_gender = f"Gender Confidence:{gender_confidence:.2f}"
        confidence_age_gender_text_age = f"Age Confidence:{age_confidence:.2f}"
        y_offset = 80 # Initial y-offset

        cv2.putText(frame, confidence_age_gender_text_gender, (fx, fy + fh + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 80), 2)
        cv2.putText(frame, confidence_age_gender_text_age, (fx, fy + fh + y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 80), 2) # Increased y-offset for the next line
    cv2.imshow('face', frame)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
cap.release()
cv2.destroyAllWindows()