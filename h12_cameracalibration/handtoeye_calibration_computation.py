import os
import glob
from typing import List, Tuple
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as SciRot
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# -------------------- HARDCODED CONFIG --------------------
DATA_DIR = "/ros2_ws/src/h12_cameracalibration/h12_cameracalibration/data/handtoeye_calibration/npzs"
assert os.path.exists(DATA_DIR), f"Data dir not found: {DATA_DIR}"
INTRINSICS_PATH = "/ros2_ws/src/h12_cameracalibration/h12_cameracalibration/data/handtoeye_calibration/intrinsics.npz"
assert os.path.exists(INTRINSICS_PATH), f"Intrinsics file not found: {INTRINSICS_PATH}"

INNER_CORNERS = (10, 7)      # (cols, rows)
SQUARE_SIZE_M = 0.010         # 1cm

HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}
from handineye_calibration_computation import visualize_r_t, stack_T, load_intrinsics_npz, target2cam_from_corners

def se3_inv(T):
    R, t = T[:3,:3], T[:3,3:4]
    Ti = np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3:4]=-R.T@t
    return Ti

def se3_mean(Ts):
    Rs = np.stack([T[:3,:3] for T in Ts], axis=0)
    ts = np.stack([T[:3, 3]  for T in Ts], axis=0)
    M = Rs.sum(axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0: R[:, -1] *= -1
    t = ts.mean(axis=0)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=t
    return T

def build_T(R,t):
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=t.reshape(3); return T

def compute_base2cam(R_base2gripper, t_base2gripper, R_cam2gripper, t_cam2gripper):
    # 1) gripper->camera
    T_cam2grip = build_T(R_cam2gripper, t_cam2gripper)
    T_grip2cam = se3_inv(T_cam2grip)
    # 2) per-frame base->camera
    T_b2c_list = [ build_T(R_bg, t_bg) @ T_grip2cam
                   for R_bg, t_bg in zip(R_base2gripper, t_base2gripper) ]
    # 3) average to one T_base2cam
    T_base2cam = se3_mean(T_b2c_list)
    return T_base2cam, T_b2c_list

def reprojection_rmse_px(
    R_base2gripper, t_base2gripper,
    R_cam2gripper, t_cam2gripper,
    corners_list, K, D,
    inner_corners, square_size_m
):
    # Object points in the board frame (Z=0). Since the board is on the gripper,
    # we can treat "world" == "board on gripper".
    cols, rows = inner_corners
    N = cols*rows
    objp = np.zeros((N,3), np.float32)
    objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2).astype(np.float32)
    objp *= float(square_size_m)

    # Get T_base2cam (avg) and helpers
    T_b2c, _ = compute_base2cam(R_base2gripper, t_base2gripper, R_cam2gripper, t_cam2gripper)
    T_c2b = se3_inv(T_b2c)
    T_g2c = se3_inv(build_T(R_cam2gripper, t_cam2gripper))  # same as above, explicit

    total_sq, total_pts = 0.0, 0
    for i in range(len(R_base2gripper)):
        T_b2g = build_T(R_base2gripper[i], t_base2gripper[i])

        # Predicted world(=board on gripper)->camera
        # T_w2c_pred = inv(T_b2c) @ T_b2g @ T_g2c
        T_w2c_pred = T_c2b @ T_b2g @ T_g2c

        R_pred = T_w2c_pred[:3,:3]
        t_pred = T_w2c_pred[:3,3:4]
        rvec, _ = cv2.Rodrigues(R_pred)
        proj, _ = cv2.projectPoints(objp, rvec, t_pred, K, D)

        diff = proj.reshape(-1,2) - corners_list[i].reshape(-1,2)
        total_sq += float(np.sum(diff*diff))
        total_pts += N

    return float(np.sqrt(total_sq/total_pts)), T_b2c

def main():
    # Load intrinsics
    K, D, distortion_model, width, height, R_rect, P = load_intrinsics_npz(INTRINSICS_PATH)
    print("[INFO] Intrinsics loaded:")
    print("K=\n", K)
    print("D=", D)
    print("distortion_model=", distortion_model)
    print(f"image size: {width} x {height}")

    # Gather samples
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {DATA_DIR}")
    npz_files = [f for f in npz_files if os.path.basename(f) != "intrinsics.npz"]
    random.shuffle(npz_files)
    print(f"[INFO] Found {len(npz_files)} NPZ files in {DATA_DIR}")
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    corners_arr = []
    rejected = 0

    for f in npz_files:
        data = np.load(f)
        corners = data["corners"]
        print(" -", os.path.basename(f))
        T_target2cam, error = target2cam_from_corners(corners, K, D)
        # T_target2cam = np.linalg.inv(T_target2cam)
        print(f"  Reprojection error rmse: {error:.3f} px")
        if error > 1.0:
            print(f"  [WARNING] High reprojection error {error:.3f} px, rejecting this sample.")
            rejected += 1
            continue
        print()
        corners_arr.append(corners)

        R_target2cam.append(T_target2cam[:3, :3])
        t_target2cam.append(T_target2cam[:3, 3].reshape(3,1))
        T_gripper2base = data["pose"]
        T_gripper2base = np.linalg.inv(T_gripper2base)
        R_gripper2base.append(T_gripper2base[:3, :3])
        t_gripper2base.append(T_gripper2base[:3, 3].reshape(3,1))

        
    print(f"[INFO] Rejected {rejected} samples due to high reprojection error.")
    assert len(R_gripper2base) == len(t_gripper2base) == len(R_target2cam) == len(t_target2cam)
    print(f"[INFO] Loaded {len(R_gripper2base)} samples")
    print(f"[INFO] Example shape: {R_gripper2base[0].shape=}, {t_gripper2base[0].shape=} {R_target2cam[0].shape=}, {t_target2cam[0].shape=}")

    # visualize_r_t(R_gripper2base, t_gripper2base, axis_len=0.05, title="Gripper->Base Poses")
    # visualize_r_t(R_target2cam, t_target2cam, axis_len=0.05, title="Target->Camera Poses")
    # plt.show()
    best_T = np.eye(4)
    best_error = float('inf')
    ransac_iters = 1000
    sample_n = 7
    for i in range(ransac_iters):

        sample_idxs = random.sample(range(len(R_gripper2base)), sample_n)
        R_gripper2base_sample = [R_gripper2base[i] for i in sample_idxs]
        t_gripper2base_sample = [t_gripper2base[i] for i in sample_idxs]
        R_target2cam_sample = [R_target2cam[i] for i in sample_idxs]
        t_target2cam_sample = [t_target2cam[i] for i in sample_idxs]

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base_sample, t_gripper2base_sample,  # lists of absolutes are expected here
            R_target2cam_sample,  t_target2cam_sample,
            method=HAND_EYE_METHODS["TSAI"]
        )
        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

        error_mm = get_error(
            R_gripper2base, t_gripper2base,
            R_target2cam,  t_target2cam,
            R_cam2gripper, t_cam2gripper,
            corners_arr, K, D
        )
        # print(f"{i}: {error_mm=:.3f} mm RMS")
        # print("\n\n")

        if error_mm < best_error:
            best_error = error_mm
            best_T = T_cam2gripper.copy()

    print(f"[RESULT] Best reprojection error: {best_error:.3f} mm RMS")


    T_gripper2base = [stack_T(R, t) for R, t in zip(R_gripper2base, t_gripper2base)]
    T_target2camera = [stack_T(R, t) for R, t in zip(R_target2cam, t_target2cam)]
    T_camera2gripper = best_T
    T_target2base = [T_g2b @ T_camera2gripper @ T_t2c for T_g2b, T_t2c in zip(T_gripper2base, T_target2camera)]
    R_list = [T[:3, :3] for T in T_target2base]
    t_stack = np.stack([T[:3, 3] for T in T_target2base], axis=0)
    visualize_r_t(R_list, t_stack, axis_len=0.005, title="Base->Target Poses")


    T_camOpt2gripper = best_T.copy()

    # ROS camera_link  -> optical frame (OpenCV) rotation
    R_opt_from_ros = np.array([[0, -1,  0],
                            [0,  0, -1],
                            [1,  0,  0]], dtype=float)
    
    T_ros2opt = np.eye(4)
    T_ros2opt[:3, :3] = R_opt_from_ros

    T_camRos2gripper = T_camOpt2gripper @ T_ros2opt



    R_ros2 = T_camRos2gripper[:3, :3]
    t_ros2 = T_camRos2gripper[:3,  3]


    # Output values
    x, y, z = t_ros2.flatten().tolist()
    qx, qy, qz, qw = SciRot.from_matrix(R_ros2).as_quat()  # xyzw
    print(f"{x=:.6f}, {y=:.6f}, {z=:.6f}")
    print(f"{qx=}, {qy=}, {qz=}, {qw=}")

    print()
    print()
    print(f"'{x}', '{y}', '{z}',")
    print(f"'{qx}', '{qy}', '{qz}', '{qw}',")

    plt.show()

if __name__ == "__main__":
    main()
