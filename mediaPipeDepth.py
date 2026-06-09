#this version works on ubuntu with the D435 and usb-c to usb-c


import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import pyrealsense2 as rs

# --- INITIALIZE MEDIAPIPE & REALSENSE ---
# (Initialization code same as previous step...)
# Assume 'detector' and 'pipeline' are already initialized

capture_count = 0

def save_capture(color_img, depth_img, depth_frame, results, count):
    """Saves images and writes landmark data to disk."""
    # Ensure directory exists
    os.makedirs("captures", exist_ok=True)
    
    # 1. Save Images
    cv2.imwrite(f"captures/color_{count}.png", color_img)
    cv2.imwrite(f"captures/depth_{count}.png", depth_img)
    
    # 2. Save NumPy & CSV
    hand_points = []
    with open("captures/landmarks.csv", "a", newline='') as f:
        writer = csv.writer(f)
        # if count == 1: # Header only on first run
        #     writer.writerow(["capture_id", "hand_index", "handedness", "landmark_idx", "x", "y", "z"])
            
        for h_idx, hand_landmarks in enumerate(results.hand_landmarks):
            # Handedness logic
            #handedness = results.multi_handedness[h_idx].classification[0].label
            handedness = results.handedness[h_idx][0].category_name

            # uncomment this if using mirrored camera input
            # if detected_hand == "Left":
            #         handedness = "Right"
            #     else:
            #         handedness = "Left"
            
            for l_idx, lm in enumerate(hand_landmarks):
                # Convert normalized coordinates --> pixel coordinates
                x_pixel = int(lm.x * color_img.shape[1])
                y_pixel = int(lm.y * color_img.shape[0])

                # Prevent out of bounds indexing
                x_pixel = np.clip(x_pixel, 0, color_img.shape[1] - 1)
                y_pixel = np.clip(y_pixel, 0, color_img.shape[0] - 1)

                # Get depth in meters 
                depth_meters = depth_frame.get_distance(x_pixel, y_pixel)
                hand_points.append([lm.x, lm.y, lm.z, depth_meters])
                writer.writerow([count, h_idx, handedness, l_idx, lm.x, lm.y, lm.z, x_pixel, y_pixel, depth_meters])
                
    np.save(f"captures/landmarks_{count}.npy", np.array(hand_points))
    print(f"Captured data set {count} saved.")

# --- 1. SETUP MEDIAPIPE ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)
detector = HandLandmarker.create_from_options(options)

# --- 2. CONNECT CAMERA ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)
colorizer = rs.colorizer()

cv2.namedWindow("RealSense: Detection")

with open("landmarks.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["capture_id", "hand_index", "handedness", "landmark_idx", "x_norm", "y_norm", "z_norm", "x_pixel", "y_pixel", "depth_meters"])

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # Convert BGR to RGB for MediaPipe
        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Detect hands
        result = detector.detect(mp_image)

        # Trigger Save
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and result.hand_landmarks:
            capture_count += 1
            save_capture(color_image, depth_image, depth_frame, result, capture_count)
        
        elif key == ord('q'):
            break

        # Draw landmarks
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * 640)
                    y = int(landmark.y * 480)
                    cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)

        # Depth visualization
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        
        combined_image = np.hstack((color_image, depth_image))
        cv2.imshow("RealSense: Detection", combined_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
