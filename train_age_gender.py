import os
import cv2
import cv2.data
import numpy as np

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- IMPROVED PATH HANDLING AND DYNAMIC REALPEOPLE LIST ---
script_dir = os.path.dirname(__file__)
path = os.path.join(script_dir, "RealPeople")

RealPeople = []
if os.path.exists(path):
    for f_name in os.listdir(path):
        full_path = os.path.join(path, f_name)
        if os.path.isdir(full_path):
            RealPeople.append(f_name)
    RealPeople.sort()
    print(f"Discovered {len(RealPeople)} people for training: {RealPeople}")
else:
    print(f"Error: 'RealPeople' directory not found at '{path}'. Please create it and add subfolders with images of people.")
    exit()

labels = []
features = []

print("Starting face data collection for training...")

if not RealPeople:
    print("No subfolders found in 'RealPeople' directory. No data to train on.")
else:
    for person_name in RealPeople:
        person_images_path = os.path.join(path, person_name)

        if not os.path.exists(person_images_path):
            print(f"Warning: Directory for '{person_name}' not found at '{person_images_path}'. Skipping.")
            continue

        label = RealPeople.index(person_name)

        print(f"Processing images for: {person_name}")
        
        image_files = [f for f in os.listdir(person_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"Warning: Directory for '{person_name}' at '{person_images_path}' is empty or contains no valid image files. Skipping.")
            continue
        
        if len(image_files) < 5:
            print(f"Notice: Only {len(image_files)} valid images found for '{person_name}'. More images generally lead to better training.")

        for img_name in image_files:
            img_path = os.path.join(person_images_path, img_name)

            try:
                im_cv = cv2.imread(img_path)
                if im_cv is None:
                    print(f"Warning: Could not read image {img_path}. It might be corrupted or an invalid image file. Skipping.")
                    continue

                gray_scale = cv2.cvtColor(im_cv, cv2.COLOR_BGR2GRAY)

                roi = face_detector.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(roi) == 0:
                    print(f"No faces found in {img_path}. Skipping.")
                    continue

                for x, y, w, h in roi:
                    face_roi = gray_scale[y:y + h, x:x + w]
                    face_resized = cv2.resize(face_roi, (100, 100))
                    features.append(face_resized)
                    labels.append(label)
                    break
            except Exception as e:
                print(f"Error reading or processing image: {img_path} - {e}")

if not features:
    print("No faces found in any images for training. Model cannot be trained.")
    print("Please ensure your 'RealPeople' subfolders contain valid images with detectable faces.")
else:
    numpy_features = np.array(features, dtype='uint8')
    label_array = np.array(labels, dtype='int')

    face_recognizer_model = cv2.face.LBPHFaceRecognizer_create()

    print(f"Training face recognition model with {len(features)} samples...")
    face_recognizer_model.train(numpy_features, label_array)

    model_filename = "face_recognizer.yaml"
    face_recognizer_model.save(model_filename)
    print(f"Model Trained and saved as {model_filename}")