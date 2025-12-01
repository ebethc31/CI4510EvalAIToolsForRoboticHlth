#import cv2
#import time

#cap = cv2.VideoCapture(0)
#target_fps = 30
#frame_delay = 1.0 / target_fps  # Time to wait between frames

#prev_frame_time = 0

#while True:
    #ret, frame = cap.read()
    #if not ret:
        #break

    # Calculate time elapsed since last frame
    #current_time = time.time()
    #time_elapsed = current_time - prev_frame_time

    # Only process and display frame if enough time has passed
   # if time_elapsed >= frame_delay:
      #  prev_frame_time = current_time

        # --- Process the frame here ---
        # For example, display it
       # cv2.imshow('Camera Feed', frame)

    #if cv2.waitKey(1) & 0xFF == ord('q'):
       # break

#cap.release()
#cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2
import random
import time
import os

# ========= CONFIG =========
webcamIndex = 0
output_path = "output_detected.mp4"
confidence_threshold = 0.4
desired_fps = 30

# Load YOLO model
yolo = YOLO("yolov8s.pt")  # or yolov8n.pt for speed

# Color generator for consistent class colors
def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

# Open webcam
cap = cv2.VideoCapture(webcamIndex)
if not cap.isOpened():
    raise Exception("Could not open video source")

# Get frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# If the camera didn't report size, read one frame to get it
if frame_width == 0 or frame_height == 0:
    ret, tmp = cap.read()
    if not ret:
        raise Exception("Failed to read a frame for size detection")
    frame_height, frame_width = tmp.shape[:2]
else:
    tmp = None

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, desired_fps, (frame_width, frame_height))

# Make sure OpenCV window receives keyboard input (important in VS Code)
cv2.namedWindow("YOLO Tracking", cv2.WINDOW_NORMAL)

frame_delay = 1.0 / desired_fps
prev_frame_time = 0.0

print("Tracking started... Press 'q' to quit.")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera disconnected or end of feed.")
        break

    # FPS control
    current_time = time.time()
    if current_time - prev_frame_time < frame_delay:
        continue
    prev_frame_time = current_time

    # YOLO detection
    results = yolo(frame, stream=True)

    for result in results:
        class_names = result.names
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = class_names[cls]
            colour = getColours(cls)

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}",
                        (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colour, 2)

    # Show output frame
    cv2.imshow("YOLO Tracking", frame)

    # Key detection
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Q pressed → stopping.")
        break

    if key == 27:  # ESC key
        print("ESC pressed → stopping.")
        break

    # Save frame to output video
    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete! Output saved to: {os.path.abspath(output_path)}")