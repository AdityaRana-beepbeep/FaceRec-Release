📌 About the Project
This project is a real-time face recognition web application built using Flask and OpenCV. It integrates LBPH face recognition with age and gender detection, allowing users to stream live video, identify people, and display their age and gender dynamically.

Initially developed as an MCA final year project, this system was designed for local face recognition using OpenCV's LBPH algorithm. To expand its functionality, the project was transformed into a fully interactive web app with Flask, enabling a user-friendly experience accessible via a browser.


🚀 Key Features

Live Video Streaming: Real-time face detection and recognition from the webcam.

Face Recognition: Uses LBPH (Local Binary Patterns Histogram) for identifying known persons.

Age & Gender Detection: Leverages OpenCV’s pre-trained deep learning models to predict age and gender.

Start/Stop Recognition: Control face detection dynamically using interactive buttons.

Responsive UI: Built with Flask and JavaScript for seamless updates.


🛠 Tech Stack

Backend: Python (Flask)

Computer Vision: OpenCV (LBPH Face Recognizer, Haar Cascade Detector)

Deep Learning Models: Pre-trained age & gender models (DNN)

Frontend: HTML, Bootstrap, JavaScript (AJAX)


🎯 How It Works

Train the Face Recognition Model using LBPH with labeled images.

Run the Flask Server (python app.py) to stream live video.

Start Face Recognition via the web interface.

Detected Person’s Name, Age & Gender are displayed dynamically.


Note: Initially, Aditya has added only his pictures in the dataset for model training, considering privacy concerns.
