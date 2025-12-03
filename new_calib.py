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

camera_matrix = np.array([[1000, 0, 640],
                          [0, 1000, 360],
                          [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.zeros(5)

# -----------------------------------------------------------
# RPY -> Rotation for uFactory 850 (extrinsic XYZ)
# -----------------------------------------------------------

def rpy_to_rot_xyz(roll, pitch, yaw):
    """Convert uFactory 850 RPY (degrees) to rotation matrix (extrinsic X→Y→Z)."""
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

    # Extrinsic rotations:
    # Final rotation = Rz(yaw) * Ry(pitch) * Rx(roll)
    # This gives the rotation of the gripper in the base frame (R_bg)
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
# Build Checkerboard 3D Points
# -----------------------------------------------------------

def build_checkerboard_object_points():
    # Create object points for the checkerboard pattern
    # Use this for calculating the pose of the checkerboard in 3D space
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    X, Y = np.meshgrid(np.arange(CHECKERBOARD[0]),
                       np.arange(CHECKERBOARD[1]))
    objp[:, 0] = X.flatten()
    objp[:, 1] = Y.flatten()
    objp *= SQUARE_SIZE
    # Returns Nx3 array of 3D points
    return objp


# -----------------------------------------------------------
# Load images + poses
# -----------------------------------------------------------

image_files = sorted(glob.glob(IMAGE_FOLDER))
print(f"Found {len(image_files)} images.")

robot_poses = load_robot_poses(POSE_FILE)
if len(robot_poses) != len(image_files):
    raise ValueError("Pose count does not match image count")

objp = build_checkerboard_object_points()

# Transformations for Hand–Eye Calibration
R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

# -----------------------------------------------------------
# Process each image
# -----------------------------------------------------------

for idx, img_file in enumerate(image_files):
    print(f"Processing image: {img_file}")

    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)
    if not found:
        print("  Checkerboard NOT found, skipping.")
        continue
    
    # Corner location refinement
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    )

    # SolvePnP (object -> camera) 
    # solvePnP X_cam = R_cam_obj * X_obj + t_cam_obj -> this gives us the values for rvec and tvec
    # r_vec and t_vec represent the transformation from object to camera (rotation and translation)
    _, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    R_cam_obj, _ = cv2.Rodrigues(rvec)

    # Convert camera->object into object->camera
    # To do this we need to invert the camera to object transformation
    R_obj_cam = R_cam_obj.T
    t_obj_cam = -R_cam_obj.T @ tvec

    R_target2cam.append(R_obj_cam)
    t_target2cam.append(t_obj_cam)

    # Robot pose (base -> gripper)
    x, y, z, roll, pitch, yaw = robot_poses[idx]

    # R_bg = Gripper rotation in base frame
    R_bg = rpy_to_rot_xyz(roll, pitch, yaw) 
    # t_bg = Gripper translation in base frame
    t_bg = np.array([[x/1000], [y/1000], [z/1000]])  # convert mm→m

    # Invert for OpenCV (gripper -> base)
    R_g2b = R_bg.T
    t_g2b = -R_bg.T @ t_bg

    R_gripper2base.append(R_g2b)
    t_gripper2base.append(t_g2b)


# -----------------------------------------------------------
# Hand–Eye Calibration
# -----------------------------------------------------------

print("\nRunning Hand–Eye Calibration...")

# AX = BX
# A = R_gripper2base, t_gripper2base
# B = R_target2cam, t_target2cam
# X is the unknown transform from camera to gripper
# Solve for X (R_cam2gripper, t_cam2gripper)
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

print("\n=== Final Camera-to-Gripper Transform ===")
print("Rotation matrix:\n", R_cam2gripper)
print("\nTranslation (mm):\n", (t_cam2gripper * 1000).flatten())

np.savez("handeye_result.npz", R=R_cam2gripper, t=t_cam2gripper)
print("\nSaved handeye_result.npz")


print("Robot pose R_bg:\n", R_bg)
print("Robot pose t_bg (m):", t_bg.flatten())
print("Gripper->base R_g2b:\n", R_g2b)
print("Gripper->base t_g2b (m):", t_g2b.flatten())

print("Determinant:", np.linalg.det(R_cam2gripper))
print("Orthogonality error:", np.linalg.norm(R_cam2gripper @ R_cam2gripper.T - np.eye(3)))


# Optional: Undistort an example image
example_img = cv2.imread(image_files[5])
h, w = example_img.shape[:2]
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
undistorted = cv2.undistort(example_img, camera_matrix, dist_coeffs, None, new_camera_mtx)
cv2.imshow("Original vs Undistorted", np.hstack([example_img, undistorted]))
cv2.waitKey(0)
cv2.destroyAllWindows()




