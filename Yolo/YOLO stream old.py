from ultralytics import YOLO
import cv2
import random
import os
import time

import cv2
import random
from ultralytics import YOLO

# ========== CONFIG ==========
webcamIndex = 0  
output_path = "output_detected.mp4"
confidence_threshold = 0.4
desired_fps = 30

# Load YOLO model
yolo = YOLO("yolov8s.pt")  # you can use yolov8n.pt for faster processing

# Color generator for consistent class colors
def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

# Open video
# Read from webcam
cap = cv2.VideoCapture(webcamIndex)  
if not cap.isOpened():
    raise Exception("Could not open video")
cap.set(cv2.CAP_PROP_FPS, desired_fps)


frame_delay = 1.0 / desired_fps  # Time to wait between frames

prev_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate time elapsed since last frame
    current_time = time.time()
    time_elapsed = current_time - prev_frame_time

    # Only process and display frame if enough time has passed
    if time_elapsed >= frame_delay:
        prev_frame_time = current_time

        # --- Process the frame here ---
        # For example, display it
        cv2.imshow('Camera Feed', frame)

frame_count = 0
print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}", end="\r")

    # # Run YOLOv8 detection
    # results = yolo(frame, stream=True)

    # # Draw boxes and labels
    # for result in results:
    #     class_names = result.names
    #     # print(result.boxes)
    #     # break 

    #     for box in result.boxes:
    #         conf = float(box.conf[0])
    #         if conf < confidence_threshold:
    #             continue

    #         x1, y1, x2, y2 = map(int, box.xyxy[0])
    #         cls = int(box.cls[0])
    #         class_name = class_names[cls]
    #         colour = getColours(cls)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    #         cv2.putText(frame, f"{class_name} {conf:.2f}",
    #                     (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.6, colour, 2)
            
    #     # Wait for 1 millisecond and check for key press
    # k = cv2.waitKey(0) & 0xFF

    # # If 'q' is pressed, break the loop
    # if k == 27: #ord('q'):
    #     print("Exited because q was pressed")
    #     break

    # out.write(frame)
    cv2.imshow("Output", frame)

cap.release()
#out.release()
cv2.destroyAllWindows()

print(f"\n Processing complete! Output saved to: {output_path}")


