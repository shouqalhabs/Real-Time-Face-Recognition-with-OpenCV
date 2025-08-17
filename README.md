# Real-Time Face Recognition with OpenCV

This project performs real-time face recognition using a webcam and the LBPH (Local Binary Patterns Histograms) algorithm. It detects faces, recognizes known individuals, and displays their names live on screen.

---

## Features

- Real-time face detection and recognition via webcam
- Trained on custom dataset of labeled face images
- Displays recognized names above detected faces
- Uses OpenCV's Haar cascades and LBPH recognizer

---

## Requirements

haarcascade_frontalface_default.xml file 


 and Make sure you have the following installed:

```bash
pip install opencv-contrib-python numpy
```

---

## Dataset Structure

Place your training images inside a folder named dataset/, organized by person name.

---

## How It Works

Training: Loads grayscale face images from dataset/, detects faces, and trains an LBPH recognizer.

Recognition: Captures frames from webcam, detects faces, and predicts identity using the trained model.

Display: Draws rectangles around faces and shows the recognized name.

---

## Test resuilts
[face recog. test]()
