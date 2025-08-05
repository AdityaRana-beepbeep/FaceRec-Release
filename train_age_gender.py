import os
import cv2
import cv2.data
import numpy as np

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

RealPeople = ['Aditya'] # Replace with actual names, e.g., ['Aditya', 'Jane Doe']

# **UPDATED PATH:** Location of your folder where subfolders for each person's images are
path = r"C:\Users\adity\OneDrive\Desktop\Project\FaceRec Release\RealPeople" 

labels = []
features = []

print("Starting face data collection for training...")

for person_name in RealPeople:
    person_images_path = os.path.join(path, person_name)
    
    # Check if the person's directory exists
    if not os.path.exists(person_images_path):
        print(f"Warning: Directory for '{person_name}' not found at '{person_images_path}'. Skipping.")
        continue

    label = RealPeople.index(person_name)

    print(f"Processing images for: {person_name}")
    for img_name in os.listdir(person_images_path):
        img_path = os.path.join(person_images_path, img_name)

        try:
            im_cv = cv2.imread(img_path)
            if im_cv is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue

            gray_scale = cv2.cvtColor(im_cv, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            # Lower scaleFactor (e.g., 1.1) and higher minNeighbors (e.g., 5) for better detection
            roi = face_detector.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(roi) == 0:
                print(f"No faces found in {img_path}. Skipping.")
                continue

            for x, y, w, h in roi:
                face_roi = gray_scale[y:y + h, x:x + w]
                face_resized = cv2.resize(face_roi, (100, 100)) # Resize for consistency
                features.append(face_resized) # Corrected typo here
                labels.append(label)         # Corrected typo here
        except Exception as e:
            print(f"Error reading or processing image: {img_path} - {e}")

if not features:
    print("No faces found in any images. Model cannot be trained. Please check your image folder and face detection.")
else:
    numpy_features = np.array(features, dtype='uint8')
    label_array = np.array(labels, dtype='int')

    make_model = cv2.face.LBPHFaceRecognizer_create()

    # Train the model
    print("Training face recognition model...")
    make_model.train(numpy_features, label_array)

    # Save the trained model. This file will be saved in the same directory as train_age_gender.py
    # Make sure this matches where recognize_age_gender.py and app.py expect it.
    model_filename = "face_recogonizer.yml" 
    make_model.save(model_filename)
    print(f"Model Trained and saved as {model_filename}")