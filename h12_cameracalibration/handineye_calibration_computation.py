import os
from typing import List
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as SciRot
import random
import matplotlib.pyplot as plt
import os
from utils import stack_T, visualize_r_t, load_intrinsics_npz, load_data, inv_SE3


def get_error(
    R_gripper2base: List[np.ndarray],
    t_gripper2base: List[np.ndarray],
    R_target2camera: List[np.ndarray],
    t_target2camera: List[np.ndarray],
    R_camera2gripper: np.ndarray,
    t_camera2gripper: np.ndarray,
    corners_list: List[np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
    inner_corners, square_size_m
) -> float:
    """
    Compute overall reprojection RMSE [px] for a hand-eye solution.

    Inputs:
        R_gripper2base, t_gripper2base : lists of gripper->base rotations/translations (each 3x3, 3x1)
        R_target2camera, t_target2camera : lists of target->camera rotations/translations (each 3x3, 3x1)
        R_camera2gripper, t_camera2gripper : camera->gripper rotation/translation (3x3, 3x1) = X
        corners_list : list of detected chessboard corners, each (N,1,2)
        K, D         : camera intrinsics and distortion
    Returns:
        RMSE reprojection error over all frames & corners [pixels].
    """
    # --- Build transforms ---
    T_gripper2base = [stack_T(R, t) for R, t in zip(R_gripper2base, t_gripper2base)]
    T_target2camera = [stack_T(R, t) for R, t in zip(R_target2camera, t_target2camera)]
    T_camera2gripper= stack_T(R_camera2gripper, t_camera2gripper)
    T_gripper2camera = inv_SE3(T_camera2gripper)

    # --- Target->Base per frame ---
    T_target2base = [T_g2b @ T_camera2gripper @ T_t2c for T_g2b, T_t2c in zip(T_gripper2base, T_target2camera)]

    # ---Mean target->base pose as the reference ---
    R_list = [T[:3, :3] for T in T_target2base]
    t_stack = np.stack([T[:3, 3] for T in T_target2base], axis=0)
    # visualize_r_t(R_list, t_stack, axis_len=0.005, title="Base->Target Poses")
    t_ref = t_stack.mean(axis=0)

    R_ref = SciRot.from_matrix(np.stack(R_list)).mean().as_matrix()

    T_ref_target2base = stack_T(R_ref, t_ref.reshape(3, 1))

    # --- Prepare chessboard model points (Z=0 plane in target frame) ---
    cols, rows = inner_corners
    N = cols * rows
    objp = np.zeros((N, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    objp *= float(square_size_m)  # meters
    
    # --- Compute reference board points in base frame (meters) ---
    objp_h = np.hstack([objp, np.ones((N, 1), dtype=np.float32)])  # (N,4)
    P_ref_b = (T_ref_target2base @ objp_h.T).T[:, :3]  # (N,3) meters

    # --- For each frame: transform board points to base, compare to reference ---
    total_sq_err_m2 = 0.0
    total_pts = 0

    for T_t2b in T_target2base:
        P_b = (T_t2b @ objp_h.T).T[:, :3]  # (N,3) meters
        d = P_b - P_ref_b                  # (N,3) meters
        total_sq_err_m2 += float(np.sum(d ** 2))
        total_pts += N

    # RMSE in millimeters
    rmse_mm = 1000.0 * float(np.sqrt(total_sq_err_m2 / total_pts))
    return rmse_mm

def calibrate_handineye(data_dir, intrinsics_path, extrinsics_path, inner_corners, square_size_m):
    # Load intrinsics
    K, D, distortion_model, width, height, R_rect, P = load_intrinsics_npz(intrinsics_path)
    rs2optical = np.load(extrinsics_path, allow_pickle=True)["cam2optical"]
    print("[INFO] Intrinsics loaded:")
    print("K=\n", K)
    print("D=", D)
    print("distortion_model=", distortion_model)
    print(f"image size: {width} x {height}")

    R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, corners_arr = load_data(data_dir, K, D, inner_corners, square_size_m)

    best_T = np.eye(4)
    best_error = float('inf')
    ransac_iters = 100
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
            method= cv2.CALIB_HAND_EYE_TSAI
        )
        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

        error_mm = get_error(
            R_gripper2base, t_gripper2base,
            R_target2cam,  t_target2cam,
            R_cam2gripper, t_cam2gripper,
            corners_arr, K, D, inner_corners, square_size_m
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

    T_cam2gripper = T_camOpt2gripper @ rs2optical
    

    R_final = T_cam2gripper[:3, :3]
    t_final = T_cam2gripper[:3,  3]


    # Output values
    x, y, z = t_final.flatten().tolist()
    qx, qy, qz, qw = SciRot.from_matrix(R_final).as_quat()  # xyzw
    print(f"{x=:.6f}, {y=:.6f}, {z=:.6f}")
    print(f"{qx=}, {qy=}, {qz=}, {qw=}")

    print()
    print()
    print(f"'{x}', '{y}', '{z}',")
    print(f"'{qx}', '{qy}', '{qz}', '{qw}',")

    plt.show()
    return T_camOpt2gripper

if __name__ == "__main__":
    INNER_CORNERS = (10, 7)      # (cols, rows)
    SQUARE_SIZE_M = 0.020         # 2cm
    file_location = os.path.dirname(os.path.abspath(__file__))
    print(f"File location: {file_location}")
    DATA_DIR = os.path.join(file_location, "data", "handineye_calibration", "npzs")
    assert os.path.exists(DATA_DIR), f"Data dir not found: {DATA_DIR}"
    INTRINSICS_PATH = os.path.join(file_location, "data", "handineye_calibration", "intrinsics.npz")
    assert os.path.exists(INTRINSICS_PATH), f"Intrinsics file not found: {INTRINSICS_PATH}"
    EXTRINSICS_PATH = os.path.join(file_location, "data", "handineye_calibration", "extrinsics.npz")
    assert os.path.exists(EXTRINSICS_PATH), f"Extrinsics file not found: {EXTRINSICS_PATH}"
    calibrate_handineye(DATA_DIR, INTRINSICS_PATH, EXTRINSICS_PATH, INNER_CORNERS, SQUARE_SIZE_M)
