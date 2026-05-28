'''
# THIS WORKS FOR STATIC INPUT #
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image



# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
#image = mp.Image.create_from_file("hand.jpg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
#cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


# THIS WORKS FOR LIVE INPUT #

import cv2
import mediapipe as mp
import numpy as np
import csv
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Create Data folder
data_folder = "Data"
os.makedirs(data_folder, exist_ok=True)

# Load hand model
base_options = python.BaseOptions(
    model_asset_path='hand_landmarker.task'
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

# Webcam
cap = cv2.VideoCapture(0)

video_running = True
paused_frame = None
last_result = None

# Stores all saved hand landmarks
saved_landmarks = []

# Capture counter
capture_id = 0

# Create CSV file + header
csv_filename = "hand_landmarks.csv"

with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "capture_id",
        "hand_index",
        "landmark_index",
        "x",
        "y",
        "z"
    ])

while True:

    if video_running:
        success, frame = cap.read()

        if not success:
            break

        # Mirror image
        frame = cv2.flip(frame, 1)

        # Save frame for pause screen
        paused_frame = frame.copy()

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hands
        result = detector.detect(mp_image)
        last_result = result

        height, width, _ = frame.shape

        # Draw landmarks
        if result.hand_landmarks:

            for hand_index, hand_landmarks in enumerate(result.hand_landmarks):

                # Draw points
                for landmark in hand_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw handedness text
                handedness = result.handedness[hand_index][0].category_name
                if detected_hand == "Left":
                    handedness = "Right"
                else:
                    handedness = "Left"

                x0 = int(hand_landmarks[0].x * width)
                y0 = int(hand_landmarks[0].y * height)

                cv2.putText(
                    frame,
                    handedness,
                    (x0, y0 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    else:
        # Show frozen frame
        frame = paused_frame.copy()

        cv2.putText(
            frame,
            "PAUSED - Press 's' to resume",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Hand Tracking", frame)

    key = cv2.waitKey(1) & 0xFF

    # Pause / Resume with 's'
    if key == ord('s'):

        # Save landmarks ONLY when pausing
        if video_running and last_result and last_result.hand_landmarks:

            current_capture = []

            # Save image to Data folder
            image_filename = os.path.join(
                data_folder,
                f"capture_{capture_id}.jpg"
            )

            cv2.imwrite(image_filename, paused_frame)

            # Save landmarks to CSV
            with open(csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)

                for hand_index, hand_landmarks in enumerate(last_result.hand_landmarks):

                    hand_points = []

                    for landmark_index, landmark in enumerate(hand_landmarks):

                        # Save to NumPy structure
                        hand_points.append([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])

                        # Save to CSV
                        writer.writerow([
                            capture_id,
                            hand_index,
                            hand_label,
                            landmark_index,
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])

                    current_capture.append(hand_points)

            saved_landmarks.append(current_capture)

            print(f"Saved capture #{capture_id}")
            print(f"Image: {image_filename}")

            capture_id += 1

        # Toggle pause/resume
        video_running = not video_running

    # Quit with 'q'
    elif key == ord('q'):
        break

# Save landmarks to NumPy file
numpy_filename = os.path.join(data_folder, f"capture.npy")
np.save(
    numpy_filename,
    np.array(saved_landmarks, dtype=object)
)

print(f"\nSaved {len(saved_landmarks)} captures")
print("NumPy file: hand_landmarks.npy")
print("CSV file: hand_landmarks.csv")
print("Images folder: Data/")

cap.release()
cv2.destroyAllWindows()