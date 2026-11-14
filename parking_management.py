import cv2
import numpy as np
from ultralytics import YOLO

# First run downloads the pretrained YOLOv8 nano model automatically.
model = YOLO("yolov8n.pt")

# Example parking-slot polygons.
# IMPORTANT: Adjust these coordinates to match your own camera view.
PARKING_SLOTS = [
    [(50, 100), (150, 100), (150, 220), (50, 220)],
    [(170, 100), (270, 100), (270, 220), (170, 220)],
    [(290, 100), (390, 100), (390, 220), (290, 220)],
    [(410, 100), (510, 100), (510, 220), (410, 220)],
]

# COCO class IDs: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


def main():
    cap = cv2.VideoCapture(0)  # Change 0 to a video file path if needed.

    if not cap.isOpened():
        print("Error: Could not open camera. Check camera permissions or connection.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read a frame from the camera.")
                break

            results = model(frame, verbose=False)
            vehicle_centers = []

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id in VEHICLE_CLASS_IDS and confidence >= 0.35:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        vehicle_centers.append((center_x, center_y))

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(
                            frame, f"Vehicle {confidence:.2f}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                        )

            occupied = 0
            available = 0

            for slot in PARKING_SLOTS:
                polygon = np.array(slot, dtype=np.int32)
                is_occupied = any(
                    cv2.pointPolygonTest(polygon, center, False) >= 0
                    for center in vehicle_centers
                )

                if is_occupied:
                    color = (0, 0, 255)
                    status = "OCCUPIED"
                    occupied += 1
                else:
                    color = (0, 255, 0)
                    status = "AVAILABLE"
                    available += 1

                cv2.polylines(frame, [polygon], True, color, 2)
                x, y = slot[0]
                cv2.putText(
                    frame, status, (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

            cv2.putText(
                frame, f"Occupied: {occupied}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            cv2.putText(
                frame, f"Available: {available}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

            cv2.imshow("Parking Management System - Machine Learning", frame)

            # Press Q to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
