import cv2
import os
import requests

# Removed unused TensorFlow Keras imports as per discussion
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array

PersonNames = ['Aditya'] # Names of people your model was trained on
cap = cv2.VideoCapture(0) # 0 for default webcam

# Load the trained face recognition model
my_model = cv2.face.LBPHFaceRecognizer_create()
model_path = "face_recogonizer.yml"
if os.path.exists(model_path):
    my_model.read(model_path)
    print(f"Loaded face recognizer model from {model_path}")
else:
    print(f"Error: Face recognizer model '{model_path}' not found. Please run train_age_gender.py first.")
    exit() # Exit if model is not found

# Load the Haar Cascade for face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Age and Gender Detection Model URLs (UPDATED!) ---
# These are more stable URLs from the spmallick/learnopencv repository
age_deploy_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
age_net_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_net.caffemodel"
gender_deploy_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
gender_net_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_net.caffemodel"

age_deploy_file = "age_deploy.prototxt"
age_net_file = "age_net.caffemodel"
gender_deploy_file = "gender_deploy.prototxt"
gender_net_file = "gender_net.caffemodel"

# Function to download files if they don't exist
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Downloaded {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename} from {url}: {e}")
            print("Please check your internet connection or the URL.")
            exit() # Exit if models can't be downloaded
    else:
        print(f"{filename} already exists.")

# Download the model files if they don't exist
download_file(age_deploy_url, age_deploy_file)
download_file(age_net_url, age_net_file)
download_file(gender_deploy_url, gender_deploy_file)
download_file(gender_net_url, gender_net_file)

# --- Load Age and Gender Detection Models ---
try:
    age_net = cv2.dnn.readNet(age_deploy_file, age_net_file)
    gender_net = cv2.dnn.readNet(gender_deploy_file, gender_net_file)
except cv2.error as e:
    print(f"Error loading DNN models: {e}")
    print("Ensure the model files were downloaded correctly and are not corrupted.")
    exit()

MODEL_MEAN_VALUES = (78.4263377603, 87.901088945, 114.5965258849)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(33-37)', '(38-43)', '(44-47)', '(48-53)', '(60-100)'] # Corrected a missing comma
gender_list = ['Male', 'Female']

# Create a named window with normal flag to allow resizing and fullscreen
cv2.namedWindow('Face Recognition & Age/Gender', cv2.WINDOW_NORMAL)
# Set the window to fullscreen
cv2.setWindowProperty('Face Recognition & Age/Gender', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Starting webcam feed...")
while True:
    ret, frame = cap.read() # x was removed as it's not used
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_detector.detectMultiScale(gray_img, 1.4, 3)

    for fx, fy, fw, fh in face:
        face_roi_gray = gray_img[fy:fy + fh, fx:fx + fw] # Grayscale ROI for LBPH
        face_roi_color = frame[fy:fy + fh, fx:fx + fw]   # Color ROI for age/gender DNN

        # --- Face Recognition ---
        label, confidence = my_model.predict(face_roi_gray)
        
        # Lower confidence values mean a more reliable match for LBPH
        confidence_threshold = 100 # You can tweak this value. Lower means stricter.
        
        name_text = "Unknown"
        if confidence < confidence_threshold:
            # Recognized
            name_text = PersonNames[label]
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (10, 255, 10), 3) # Green for recognized
            cv2.putText(frame, name_text, (fx, fy - 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 20), 2)
            cv2.putText(frame, f"Conf: {round(confidence, 2)}", (fx, fy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 20), 2)
        else:
            # Unrecognized
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 3) # Red for unknown
            cv2.putText(frame, "Unknown", (fx, fy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # --- Age and Gender Prediction ---
        # Create a blob from the color face ROI
        blob = cv2.dnn.blobFromImage(face_roi_color, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

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

        # Display Age and Gender
        age_gender_text = f"{gender}, {age}"
        cv2.putText(frame, age_gender_text, (fx, fy + fh + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 20), 2)
        
        # Display confidences
        confidence_age_gender_text_gender = f"G Conf:{gender_confidence:.2f}"
        confidence_age_gender_text_age = f"A Conf:{age_confidence:.2f}"
        
        cv2.putText(frame, confidence_age_gender_text_gender, (fx, fy + fh + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 80), 2)
        cv2.putText(frame, confidence_age_gender_text_age, (fx, fy + fh + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 80), 2)
    
    cv2.imshow('Face Recognition & Age/Gender', frame)

    # Press 'q' to exit the loop (changed from 'e' for common convention)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()