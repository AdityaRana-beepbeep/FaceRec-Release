import os
import cv2
import cv2.data
import numpy as np

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

RealPeople = ['Aditya'] # Replace with actual names

#Location of your folder
path = r"C:\Users\adity\OneDrive\Desktop\Project\FaceRec\RealPeople" 

labels = []
features = []

for person_name in RealPeople:
    person_images_path = os.path.join(path, person_name)
    label = RealPeople.index(person_name)

    for img_name in os.listdir(person_images_path):
        img_path = os.path.join(person_images_path, img_name)

        try:
            im_cv = cv2.imread(img_path)
            gray_scale = cv2.cvtColor(im_cv, cv2.COLOR_BGR2GRAY)

            roi = face_detector.detectMultiScale(gray_scale, 1.4, 3)

            for x, y, w, h in roi:
                face_roi = gray_scale[y:y + h, x:x + w]
                face_resized = cv2.resize(face_roi, (100, 100))
                features.append(face_resized)
                labels.append(label)  
        except Exception as e:
            print(f"Error reading or processing image: {img_path} - {e}")

numpy_features = np.array(features, dtype='uint8')
label_array = np.array(labels, dtype='int')

make_model = cv2.face.LBPHFaceRecognizer_create()

make_model.train(numpy_features, label_array)

make_model.save("face_recogonizer.yml")
print("Model Trained")