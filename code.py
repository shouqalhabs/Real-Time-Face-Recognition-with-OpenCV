import cv2
import os
import numpy as np

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 1: Prepare training data
def prepare_training_data(data_folder_path):
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for person_name in os.listdir(data_folder_path):
        person_path = os.path.join(data_folder_path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person_name

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            faces_rect = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces_rect:
                face = image[y:y+h, x:x+w]
                faces.append(face)
                labels.append(label_id)

        label_id += 1

    return faces, labels, label_map

# Step 2: Train recognizer
faces, labels, label_map = prepare_training_data("dataset")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Step 3: Real-time recognition from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        face = gray[y:y+h, x:x+w]
        label_id, confidence = recognizer.predict(face)
        label_text = label_map[label_id]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
