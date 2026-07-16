#this version works on ubuntu with the D435 and usb-c to usb-c

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import pyrealsense2 as rs
import math

# --- INITIALIZE MEDIAPIPE & REALSENSE ---
# (Initialization code same as previous step...)
# Assume 'detector' and 'pipeline' are already initialized

capture_count = 0
last_command = None

# Robustness testing
robustness_mode = False
print(f"Robustness mode starts as: {robustness_mode}")
expected_command = "stop"

# Safety filtering
give_frame_threshold = 60
give_counter = 0
robot_move = False
last_robot_state = False

# Agreement variables 
required_give_frames = 54
allowed_stop_frames = 6
give_frames = 0
stop_frames = 0

# Statistics
give_attempts = 0
successful_gives = 0

stop_attempts = 0
successful_stops = 0

false_gives = 0
false_stops = 0

print_counter = 0

def save_capture(color_img, depth_img, depth_frame, depth_intrinsics, results, count):
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
            
            gesture = detect_hand_gesture(hand_landmarks. handedness)

            for l_idx, lm in enumerate(hand_landmarks):
                # Convert normalized coordinates --> pixel coordinates
                x_pixel = int(lm.x * color_img.shape[1])
                y_pixel = int(lm.y * color_img.shape[0])

                # Prevent out of bounds indexing
                x_pixel = np.clip(x_pixel, 0, color_img.shape[1] - 1)
                y_pixel = np.clip(y_pixel, 0, color_img.shape[0] - 1)

                # Get depth in meters 
                depth_meters = depth_frame.get_distance(x_pixel, y_pixel)

                # Convert pixel + depth --> 3D camera coordinates
                point_3d = rs.rs2_deproject_pixel_to_point(
                    depth_intrinsics,
                    [x_pixel, y_pixel],
                    depth_meters
                )
                x_meters = point_3d[0]
                y_meters = point_3d[1]
                z_meters = point_3d[2]

                hand_points.append([x_meters, y_meters, x_meters, depth_meters])
                writer.writerow([count, h_idx, handedness, gesture, l_idx, lm.x, lm.y, lm.z, x_pixel, y_pixel, depth_meters, x_meters, y_meters, z_meters])
                
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
profile = pipeline.get_active_profile()

depth_profile = rs.video_stream_profile(
    profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
align = rs.align(rs.stream.color)
colorizer = rs.colorizer()

cv2.namedWindow("RealSense: Detection")

with open("captures/landmarks.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["capture_id", "hand_index", "handedness", "gesture", "landmark_idx", "x_norm", "y_norm", "z_norm", "x_pixel", "y_pixel", "depth_meters", "x_meters", "y_meters", "z_meters"])

def detect_hand_gesture(hand_landmarks, handedness):
    def dist(a, b):
        return math.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2
        )

    wrist = hand_landmarks[0]

    # Determine whether palm or back of hand faces camera
    index_mcp = hand_landmarks[5]
    pinky_mcp = hand_landmarks[17]

    v1 = np.array([
        index_mcp.x - wrist.x,
        index_mcp.y - wrist.y,
        index_mcp.z - wrist.z
    ])

    v2 = np.array([
        pinky_mcp.x - wrist.x,
        pinky_mcp.y - wrist.y,
        pinky_mcp.z - wrist.z
    ])

    hand_normal = np.cross(v1, v2)

    back_of_hand = False

    # Sign flips depending on handedness
    if handedness == "Right":
        if hand_normal[2] > 0:
            back_of_hand = True

    elif handedness == "Left":
        if hand_normal[2] < 0:
            back_of_hand = True

    # Finger extension detection
    finger_pairs = [
        (8, 6), # index
        (12, 10), # middle
        (16, 14), # ring
        (20, 18) # pinky
    ]

    open_fingers = 0

    for tip_idx, pip_idx in finger_pairs:
        tip = hand_landmarks[tip_idx]
        pip = hand_landmarks[pip_idx]

        tip_dist = dist(tip, wrist)
        pip_dist = dist(pip, wrist)

        if tip_dist > pip_dist * 1.15:
            open_fingers += 1
        
        # Thumb extension
        thumb_tip = hand_landmarks[4]
        thumb_ip = hand_landmarks[3]

        thumb_tip_dist = dist(
            thumb_tip,
            wrist
        )

        thumb_ip_dist = dist(
            thumb_ip,
            wrist
        )

        thumb_open = (
            thumb_tip_dist >
            thumb_ip_dist * 1.10
        )

        total_open = open_fingers + int(thumb_open)

    # Count folded fingers
    closed_fingers = 0

    for tip_idx, pip_idx in finger_pairs:

        tip = hand_landmarks[tip_idx]
        pip = hand_landmarks[pip_idx]

        if dist(tip, wrist) < dist(pip, wrist) * 1.05:
            closed_fingers += 1

    if closed_fingers >= 3:
        return "CLOSED HAND"
    
    # Open hand with back facing camera --> stop
    if back_of_hand and open_fingers >= 3:
        return "BACK OF HAND"

    if open_fingers >= 3:
        # Knuckle landmarks 
        index_mcp = hand_landmarks[5]
        pinky_mcp = hand_landmarks[17]

        # Direction vector
        dx = pinky_mcp.x - index_mcp.x 
        dy = pinky_mcp.y - index_mcp.y

        epsilon = 1e-6
        horizontal_ratio = abs(dx) / (abs(dy) + epsilon)
        vertical_ratio = abs(dy) / (abs(dx) + epsilon)

        # Fingers mostly horizontal, thumb up
        if horizontal_ratio > 1.2:
            return "OPEN HAND THUMB SIDE" # Code seems to get thumb and side mixed up, so I just switched the labels
        # Fingers mostly vertical, thumb sideways
        if vertical_ratio > 1.2:
            return "OPEN HAND THUMB UP"
        return "OPEN HAND"
    return "UNKNOWN"

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
            save_capture(color_image, depth_image, depth_frame, depth_intrinsics, result, capture_count)

        if key == ord('r'):
            robustness_mode = not robustness_mode

            if robustness_mode:
                print("\nROBUSTNESS TEST ENABLED")
            else:
                print("\nROBUSTNESS TEST DISABLED")

        elif key == ord('g'):
            expected_command = "give"
            give_frames = 0
            stop_frames = 0
            robot_move = False
            print("\nTesting GIVE command")

        elif key == ord('t'):
            expected_command = "stop"
            give_frames = 0
            stop_frames = 0
            robot_move = False
            print("\nTesting STOP command")
       
        elif key == ord('q'):
            break

        # Draw landmarks
        if result.hand_landmarks:
            for h_idx, hand_landmarks in enumerate(
                result.hand_landmarks):

                # Detect gesture
                handedness = result.handedness[h_idx][0].category_name
                gesture = detect_hand_gesture(hand_landmarks, handedness)

                # Safety-first command logic
                if gesture.startswith("OPEN HAND"):
                    current_command = "give"
                    give_frames += 1
                else:
                    current_command = "stop"
                    stop_frames += 1
                    give_frames = max(0, give_frames - 1)

                # Robot movement authorization
                if give_frames >= required_give_frames and stop_frames <= allowed_stop_frames:
                    robot_move = True

                # Too many STOP frames invalidates the authorization attempt
                if stop_frames > allowed_stop_frames:
                    give_frames = 0
                    stop_frames = 0
                    robot_move = False
                    
                    if robot_move and not last_robot_state:
                        print("\nROBOT MOVEMENT AUTHORIZED")
                        last_robot_state = True
                    elif not robot_move and last_robot_state:
                        print("\nROBOT MOVEMENT REVOKED")
                        last_robot_state = False
                
                if robustness_mode:
                    if expected_command == "give":
                        give_attempts += 1
                        if current_command == "give":
                            successful_gives += 1
                        else:
                            false_stops += 1
                    elif expected_command == "stop":
                        stop_attempts += 1
                        if current_command == "stop":
                            successful_stops += 1
                        else: false_gives += 1

                # Print only when command changes
                if current_command != last_command:
                    print(current_command)
                    last_command = current_command

            for hand_landmarks in result.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * 640)
                    y = int(landmark.y * 480)
                    cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)
                
                # Wrist position label
                wrist = hand_landmarks[0]

                wrist_x = int(float(wrist.x) * color_image.shape[1])
                wrist_y = int(float(wrist.y) * color_image.shape[0])

                text_position = (int(wrist_x), int(wrist_y - 20))

                cv2.putText(
                    color_image,
                    str(gesture),
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                # Wrist depth
                wrist_px = np.clip(wrist_x, 0, color_image.shape[1] - 1)
                wrist_py = np.clip(wrist_y, 0, color_image.shape[0] - 1)

                wrist_depth = depth_frame.get_distance(wrist_px, wrist_py)

                # Landmark 9 (middle finger MCP joint)
                lm9 = hand_landmarks[9]

                lm9_x_pixel = int(lm9.x * color_image.shape[1])
                lm9_y_pixel = int(lm9.y * color_image.shape[0])

                # Prevent out-of-bounds indexing
                lm9_x_pixel = np.clip(
                    lm9_x_pixel,
                    0,
                    color_image.shape[1] - 1
                )

                lm9_y_pixel = np.clip(
                    lm9_y_pixel,
                    0,
                    color_image.shape[0] - 1
                )

                # Depth at landmark 9
                lm9_depth = depth_frame.get_distance(
                    lm9_x_pixel,
                    lm9_y_pixel
                )

                # Convert pixel + depth to 3D camera coordinates
                lm9_3d = rs.rs2_deproject_pixel_to_point(
                    depth_intrinsics,
                    [lm9_x_pixel, lm9_y_pixel],
                    lm9_depth
                )

                lm9_x_m = lm9_3d[0]
                lm9_y_m = lm9_3d[1]
                lm9_z_m = lm9_3d[2]

                print_counter += 1
                if print_counter % 15 == 0:
                    print(
                        f"LM9: "
                        f"X={lm9_x_m:.3f} m, "
                        f"Y={lm9_y_m:.3f} m, "
                        f"Z={lm9_z_m:.3f} m"
                    )

                cv2.putText(
                    color_image,
                    f"Depth: {wrist_depth:.3f} m",
                    (wrist_x, wrist_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
        else:
            current_command = "stop"

            if current_command != last_command:
                print(current_command)
                last_command = current_command

        # Depth visualization
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        
        if robustness_mode:

            cv2.putText(
                color_image,
                f"Expected: {expected_command}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2
            )

            cv2.putText(
                color_image,
                f"Predicted: {current_command}",
                (10,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2
            )

            cv2.putText(
                color_image,
                f"False Gives: {false_gives}",
                (10,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )

            cv2.putText(
                color_image,
                f"GIVE Frames: {give_frames}/{required_give_frames}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            status_text = (
                "ROBOT CAN MOVE"
                if robot_move
                else
                "ROBOT LOCKED"
            )

            status_color = (
                (0, 255, 0)
                if robot_move
                else
                (0, 0, 255)
            )

            cv2.putText(
                color_image,
                status_text,
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )

        combined_image = np.hstack((color_image, depth_image))
        cv2.imshow("RealSense: Detection", combined_image)

finally:

    print("\n========== ROBUSTNESS RESULTS ==========")

    print(f"GIVE attempts: {give_attempts}")
    print(f"Successful GIVE: {successful_gives}")

    print(f"STOP attempts: {stop_attempts}")
    print(f"Successful STOP: {successful_stops}")

    print(f"False GIVE commands: {false_gives}")
    print(f"False STOP commands: {false_stops}")

    if give_attempts > 0:
        print(
            f"GIVE accuracy: "
            f"{100 * successful_gives / give_attempts:.2f}%"
        )

    if stop_attempts > 0:
        print(
            f"STOP accuracy: "
            f"{100 * successful_stops / stop_attempts:.2f}%"
        )

    pipeline.stop()
    cv2.destroyAllWindows()