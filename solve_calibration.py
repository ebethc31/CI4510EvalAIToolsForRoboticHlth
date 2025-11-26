import cv2
import numpy as np
import glob
import os
import pickle
import sys
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
DATA_DIR = "new_calibration_data"
CHESSBOARD_SIZE = (9, 7) #inner corners
SQUARE_SIZE = 0.019  # #measured the printed paper
# ---------------------

#calculate rotation and translation from (x,y,z,roll,pitch,yaw) saved in robot_poses.txt
def load_poses_manual_spec(pose_file):
    """
    Load poses applying Robot Manual specs:
    1. Units are mm -> Divide by 1000.0 
    2. Rotation is Extrinsic X->Y->Z ('xyz') 
    """
    if not os.path.exists(pose_file):
        print(f"Error: {pose_file} not found.")
        sys.exit(1)

    R_gripper2base = []
    t_gripper2base = []
    raw_lines = []
    
    with open(pose_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            
            vals = [float(x) for x in parts]
            raw_lines.append(vals)
            
            # RULE 1: Units are mm, convert to meters
            t_vec = np.array(vals[0:3]) / 1000.0
            
            # RULE 2: First Rotate X, then Y, then Z (Extrinsic 'xyz')
            r = R.from_euler('xyz', vals[3:], degrees=True)
            r_mat = r.as_matrix()
            
            R_gripper2base.append(r_mat)
            t_gripper2base.append(t_vec)
            
    return R_gripper2base, t_gripper2base

def main():
    print(f"=== FINAL CALIBRATION SOLVER ===")
    print(f"Specs derived from Manual:")
    print(f"  - Rotation: Extrinsic X -> Y -> Z ('xyz')")
    print(f"  - Scale:    Millimeters -> Meters (/1000.0)")
    print(f"  - Target:   {SQUARE_SIZE*1000:.1f}mm Square Size")
    print("=" * 60)
    
    # 1. Load Images
    image_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.[jJ][pP][gG]")))
    if not image_files:
        print(f"Error: No JPG images found in {DATA_DIR}.")
        return

    # 2. Process Images (Vision)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    
    img_points = []
    obj_points = []
    valid_indices = []
    
    print(f"Processing {len(image_files)} images...")
    gray_shape = None
    
    #for each image, find the chessboard corners and save the pixel coordinates
    for i, fname in enumerate(image_files):
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]
        
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_points.append(corners)
            obj_points.append(objp)
            valid_indices.append(i)
            
    if not img_points:
        print("Error: No corners detected in any image.")
        return

    # 3. Calibrate Intrinsics
    #this is the non-standard transform from camera to image
    print("Calibrating Camera Intrinsics...")
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_shape, None, None)
    print(f"   > Intrinsic Error: {ret:.4f} pixels")

    # 4. Load Robot Poses
    pose_file = os.path.join(DATA_DIR, "robot_poses.txt")
    R_g2b_all, t_g2b_all = load_poses_manual_spec(pose_file)
    
    # Sync images with poses
    if len(R_g2b_all) < len(valid_indices):
        print(f"Error: Mismatch! {len(valid_indices)} valid images but only {len(R_g2b_all)} poses.")
        return

    R_gripper2base = [R_g2b_all[i] for i in valid_indices]
    t_gripper2base = [t_g2b_all[i] for i in valid_indices]
    
    # Prepare Camera Poses for Hand-Eye
    R_target2cam = []
    t_target2cam = []
    for i in range(len(rvecs)):
        rm, _ = cv2.Rodrigues(rvecs[i])
        R_target2cam.append(rm)
        t_target2cam.append(tvecs[i])

    # 5. Solve Hand-Eye
    print("\nSolving Hand-Eye Calibration...")
    R_cal, t_cal = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, 
        R_target2cam, t_target2cam, 
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    # 6. Verify Accuracy (Consistency Check)
    errors = []
    for k in range(len(R_gripper2base)):
        T_g = np.eye(4)
        T_g[:3, :3] = R_gripper2base[k]
        T_g[:3, 3] = t_gripper2base[k]
        
        T_he = np.eye(4)
        T_he[:3, :3] = R_cal
        T_he[:3, 3] = t_cal.flatten()
        
        T_c = np.eye(4)
        T_c[:3, :3] = R_target2cam[k]
        T_c[:3, 3] = t_target2cam[k].flatten()
        
        # Calculate Base->Target
        T_result = T_g @ T_he @ T_c
        errors.append(T_result[:3, 3])
        
    std_dev = np.sqrt(np.mean(np.var(np.array(errors), axis=0))) * 1000 # mm
    
    print(f"\nRESULTS:")
    print(f"   Consistency Error: {std_dev:.3f} mm")

    # Create Result Matrix
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cal
    T_cam2gripper[:3, 3] = t_cal.flatten()

    print("cam2gripper:")
    print(T_cam2gripper)
    print("camera matrix")
    print(K)
    
    if std_dev < 10.0:
        print("   ✅ SUCCESS! Calibration is accurate.")
        
        

        # Save everything needed for runtime
        calib_data = {
            'camera_matrix': K,
            'dist_coeffs': D,
            'T_cam2gripper': T_cam2gripper,
            'chessboard_size': CHESSBOARD_SIZE,
            'square_size': SQUARE_SIZE,
            'robot_scale_factor': 1000.0,
            'robot_rot_seq': 'xyz'
        }


        
        os.makedirs("calibration_data", exist_ok=True)
        outfile = "calibration_data/robot_camera_calibration.pkl"
        with open(outfile, 'wb') as f:
            pickle.dump(calib_data, f)
        print(f"   Saved to {outfile}")
    else:
        print("   ⚠️ WARNING: Error is > 10mm.")
        print("   Did you record a new dataset with the Pyramid (Tilt) method?")

if __name__ == "__main__":
    main()