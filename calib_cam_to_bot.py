import cv2
import numpy as np
import glob
import math

# -----------------------------------------------------------
# User Parameters
# -----------------------------------------------------------

IMAGE_FOLDER = "./new_calibration_data/*.jpg"
POSE_FILE = "./new_calibration_data/robot_poses.txt"

CHECKERBOARD = (9, 7)    # inner corners (width, height)
SQUARE_SIZE = 0.019      # meters


# -----------------------------------------------------------
# STEP 1 — CAMERA CALIBRATION
# -----------------------------------------------------------

def calibrate_camera(image_folder):
    print("\n=== CAMERA CALIBRATION ===")

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    images = sorted(glob.glob(image_folder))
    if len(images) == 0:
        raise RuntimeError("No images found for camera calibration.")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11,11), (-1,-1), criteria
            )
            imgpoints.append(corners2)
            #print(f"  ✓ Chessboard detected in {fname}")
        else:
            print(f"  ✗ Chessboard NOT found in {fname}")

    print("\nCalibrating...")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("\n=== Camera Calibration Results ===")
    print("Camera Matrix K:\n", K)
    print("Distortion Coeffs:", dist.ravel())
    print("Reprojection Error:", ret)

    np.savez("camera_calibration.npz", K=K, dist=dist)
    print("\nSaved camera calibration to camera_calibration.npz\n")

    return K, dist


# -----------------------------------------------------------
# RPY -> Rotation matrix (using your existing logic)
# -----------------------------------------------------------

def rpy_to_rot_xyz(roll, pitch, yaw):
    r = math.radians(roll)
    p = math.radians(pitch)
    y = math.radians(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(r), -math.sin(r)],
        [0, math.sin(r),  math.cos(r)]
    ])

    Ry = np.array([
        [ math.cos(p), 0, math.sin(p)],
        [0,            1, 0],
        [-math.sin(p), 0, math.cos(p)]
    ])

    Rz = np.array([
        [math.cos(y), -math.sin(y), 0],
        [math.sin(y),  math.cos(y), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


# -----------------------------------------------------------
# Load robot poses
# -----------------------------------------------------------

def load_robot_poses(file):
    poses = []
    with open(file, "r") as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            vals = list(map(float, parts))
            if len(vals) != 6:
                raise ValueError(f"Invalid pose line: {line}")
            poses.append(vals)
    return poses


# -----------------------------------------------------------
# Build checkerboard object points
# -----------------------------------------------------------

def build_checkerboard_object_points():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    X, Y = np.meshgrid(np.arange(CHECKERBOARD[0]),
                       np.arange(CHECKERBOARD[1]))
    objp[:, 0] = X.flatten()
    objp[:, 1] = Y.flatten()
    objp *= SQUARE_SIZE
    return objp


# -----------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------

print("=== Starting Combined Camera + Hand–Eye Calibration ===")

# 1) Calibrate camera FIRST
camera_matrix, dist_coeffs = calibrate_camera(IMAGE_FOLDER)

# 2) Load images and robot poses
image_files = sorted(glob.glob(IMAGE_FOLDER))
robot_poses = load_robot_poses(POSE_FILE)

print(f"\nFound {len(image_files)} images.")

if len(robot_poses) != len(image_files):
    raise ValueError("Pose count does not match image count")

objp = build_checkerboard_object_points()

# Hand–eye buffers
R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

# -----------------------------------------------------------
# Process each image (PnP + robot pose)
# -----------------------------------------------------------

for idx, img_file in enumerate(image_files):
    #print(f"\nProcessing image: {img_file}")

    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)
    if not found:
        print("  ✗ Checkerboard NOT found, skipping.")
        continue
    
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))

    # SolvePnP gives camera→object, invert it to get object→camera
    _, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    R_cam_obj, _ = cv2.Rodrigues(rvec)
    R_obj_cam = R_cam_obj.T
    t_obj_cam = -R_cam_obj.T @ tvec

    R_target2cam.append(R_obj_cam)
    t_target2cam.append(t_obj_cam)

    # Robot pose: base → gripper
    x, y, z, roll, pitch, yaw = robot_poses[idx]
    R_bg = rpy_to_rot_xyz(roll, pitch, yaw)
    t_bg = np.array([[x/1000], [y/1000], [z/1000]])  # mm → m

    R_gripper2base.append(R_bg)
    t_gripper2base.append(t_bg)


# -----------------------------------------------------------
# Hand–Eye Calibration
# -----------------------------------------------------------

print("\n=== Running Hand–Eye Calibration ===")

R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

print("\n=== Final Camera-to-Gripper Transform ===")
print("Rotation:\n", R_cam2gripper)
print("Translation (mm):\n", (t_cam2gripper * 1000).flatten())

np.savez("handeye_result.npz", R=R_cam2gripper, t=t_cam2gripper)
print("\nSaved handeye_result.npz")

print("Orthogonality error:", np.linalg.norm(R_cam2gripper @ R_cam2gripper.T - np.eye(3)))

# -----------------------------------------------------------
# Optional: Undistort a sample image
# -----------------------------------------------------------

img = cv2.imread(image_files[0])
if img is None:
    print("ERROR: Image failed to load")
    exit()

h, w = img.shape[:2]

new_K, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 0)
und = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_K)

cv2.imshow("Original", img)
cv2.imshow("Undistorted", und)
cv2.waitKey(0)
