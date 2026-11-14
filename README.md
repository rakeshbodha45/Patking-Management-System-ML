Parking Management System Using Machine Learning

A prototype smart-parking application using Python, a pretrained YOLOv8 object-detection model, and OpenCV.

## Features
- Detects vehicles (cars, motorcycles, buses, and trucks) in a camera frame.
- Checks whether configured parking-slot regions contain detected vehicle centers.
- Displays occupied and available slot counts.
- Uses green outlines for available slots and red outlines for occupied slots.

## Requirements
- Python 3.9 or newer (Python version supported by the installed Ultralytics release)
- A webcam or compatible camera
- Internet access on the first run to download the pretrained `yolov8n.pt` model

## Installation
```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python parking_management.py
```

Press `Q` to close the camera window.

## Important: Configure your parking slots
The sample slot coordinates in `PARKING_SLOTS` are placeholders. Change the four corner points for every parking space to match the camera image. The camera should be positioned so the parking spaces are clearly visible.

## Notes and limitations
- This is a working starter prototype, not a complete deployed parking product.
- It uses a pretrained YOLOv8 model; it does not train a new model on a custom dataset.
- Slot occupancy is estimated by checking whether a detected vehicle's center point falls inside a slot polygon. Camera angle, overlapping vehicles, lighting, and incorrect slot coordinates can affect accuracy.
- Camera index `0` is used by default. Change `cv2.VideoCapture(0)` if your camera uses another index or if you want to read a video file.

## Technologies
Python, YOLOv8 (Ultralytics), OpenCV, NumPy.
