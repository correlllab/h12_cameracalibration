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
    R_base_gripper: List[np.ndarray],
    t_base_gripper: List[np.ndarray],
    R_camera_target: List[np.ndarray],
    t_camera_target: List[np.ndarray],
    R_gripper_camera: np.ndarray,
    t_gripper_camera: np.ndarray,
    corners_list: List[np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
    inner_corners, square_size_m
) -> float:
    """
    Compute overall reprojection RMSE [px] for a hand-eye solution.

    Inputs:
        R_base_gripper, t_base_gripper : lists of base->gripper rotations/translations (each 3x3, 3x1)
        R_camera_target, t_camera_target : lists of camera->target rotations/translations (each 3x3, 3x1)
        R_gripper_camera, t_gripper_camera : gripper->camera rotation/translation (3x3, 3x1) = X
        corners_list : list of detected chessboard corners, each (N,1,2)
        K, D         : camera intrinsics and distortion
    Returns:
        Returns:
        rmse_px: reprojection RMSE [px]
    """
    # --- Build transforms ---
    H_base_gripper = [stack_T(R, t) for R, t in zip(R_base_gripper, t_base_gripper)]
    H_camera_target = [stack_T(R, t) for R, t in zip(R_camera_target, t_camera_target)]
    H_gripper_camera = stack_T(R_gripper_camera, t_gripper_camera)

    # --- Target->Base per frame ---
    H_base_target = [H_b_g @ H_gripper_camera @ H_c_t for H_b_g, H_c_t in zip(H_base_gripper, H_camera_target)]
    # Rs = [H[:3,:3] for H in H_base_target]
    # Ts = [H[:3,3] for H in H_base_target]
    # visualize_r_t(Rs, np.stack(Ts), axis_len=0.05, title="Base->Target Poses for Error Computation", show=True)
    R_ref = SciRot.from_matrix(np.stack([H[:3,:3] for H in H_base_target])).mean().as_matrix()
    t_ref = np.mean([H[:3,3] for H in H_base_target], axis=0).reshape(3,1)
    H_base_target_ref = stack_T(R_ref, t_ref)

    # --- Prepare chessboard model points (Z=0 plane in target frame) ---
    cols, rows = inner_corners
    N = cols * rows
    objp = np.zeros((N, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    objp *= float(square_size_m)  # meters


    total_sq_error_px = 0.0
    total_corners = 0
    H_camera_gripper = inv_SE3(H_gripper_camera)
    for H_b_g, corners in zip(H_base_gripper, corners_list):
        H_g_b = inv_SE3(H_b_g)
        pred_H_c_t = H_camera_gripper @ H_g_b @ H_base_target_ref
        rvec, tvec = cv2.Rodrigues(pred_H_c_t[:3, :3])[0], pred_H_c_t[:3, 3].reshape(3, 1)

        imgpts, _ = cv2.projectPoints(
            objp,
            rvec,
            tvec,
            K,
            D
        )  # imgpts: (N,1,2)

        d = imgpts - corners  # (N,1,2)
        total_sq_error_px += float(np.sum(d ** 2))
        total_corners += corners.shape[0]
    rmse_px = float(np.sqrt(total_sq_error_px / total_corners))

    return  rmse_px

def calibrate_handineye(data_dir, intrinsics_path, extrinsics_path, inner_corners, square_size_m):
    # Load intrinsics
    K, D, distortion_model, width, height, R_rect, P = load_intrinsics_npz(intrinsics_path)
    rs2optical = np.load(extrinsics_path, allow_pickle=True)["T_camerabase_cameraoptical"]
    print("[INFO] Intrinsics loaded:")
    print("K=\n", K)
    print("D=", D)
    print("distortion_model=", distortion_model)
    print(f"image size: {width} x {height}")

    R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, corners_arr = load_data(data_dir, K, D, inner_corners, square_size_m)
    visualize_r_t(R_gripper2base, t_gripper2base)
    best_T = np.eye(4)
    best_error = float('inf')
    ransac_iters = 20
    sample_n = 5
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

        error_px = get_error(
            R_gripper2base, t_gripper2base,
            R_target2cam,  t_target2cam,
            R_cam2gripper, t_cam2gripper,
            corners_arr, K, D, inner_corners, square_size_m
        )
        print(f"{i}: {error_px=:.3f} px RMS")
        print("\n\n")

        if error_px < best_error:
            best_error = error_px
            best_T = T_cam2gripper.copy()

    print(f"[RESULT] Best reprojection error: {best_error:.3f} px RMS")


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
    INTRINSICS_PATH = os.path.join(file_location, "data", "handineye_calibration", "intrinsics_0.npz")
    assert os.path.exists(INTRINSICS_PATH), f"Intrinsics file not found: {INTRINSICS_PATH}"
    EXTRINSICS_PATH = os.path.join(file_location, "data", "handineye_calibration", "extrinsics_0.npz")
    assert os.path.exists(EXTRINSICS_PATH), f"Extrinsics file not found: {EXTRINSICS_PATH}"
    calibrate_handineye(DATA_DIR, INTRINSICS_PATH, EXTRINSICS_PATH, INNER_CORNERS, SQUARE_SIZE_M)
