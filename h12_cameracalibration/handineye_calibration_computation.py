import os
from typing import List
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as SciRot
import random
import matplotlib.pyplot as plt
import os
from utils import stack_H, visualize_r_t, load_intrinsics_npz, load_data, inv_SE3, predict_corners, visualize_corners
import scipy.optimize as opt

def get_avg_transform(H_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute average SE3 transform from a list of SE3 transforms.

    Inputs:
        H_list : list of SE3 transforms (each 4x4)

    Returns:
        H_avg : average SE3 transform (4x4)
    """
    # Compute the average rotation
    R_list = [H[:3, :3] for H in H_list]
    R_avg = SciRot.from_matrix(np.mean([SciRot.from_matrix(R).as_matrix() for R in R_list], axis=0)).as_matrix()

    # Compute the average translation
    t_list = [H[:3, 3] for H in H_list]
    t_avg = np.mean(t_list, axis=0).reshape(3, 1)

    # Stack the average rotation and translation into a single SE3 transform
    H_avg = stack_H(R_avg, t_avg)
    return H_avg

def get_base_target_ref_transform(H_base_gripper_list, H_camera_target_list, H_gripper_camera, display=False):
    # --- Target->Base per frame ---
    H_base_target_list = [H_base_gripper @ H_gripper_camera @ H_cam_target for H_base_gripper, H_cam_target in zip(H_base_gripper_list, H_camera_target_list)]
    if display:
        R_list = [T[:3, :3] for T in H_base_target_list]
        t_stack = np.stack([T[:3, 3] for T in H_base_target_list], axis=0)
        visualize_r_t(R_list, t_stack, axis_len=0.005, title="Base->Target Poses")
    H_base_target_ref = get_avg_transform(H_base_target_list)
    return H_base_target_ref


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
    H_base_gripper = [stack_H(R, t) for R, t in zip(R_base_gripper, t_base_gripper)]
    H_camera_target = [stack_H(R, t) for R, t in zip(R_camera_target, t_camera_target)]
    H_gripper_camera = stack_H(R_gripper_camera, t_gripper_camera)

    
    H_base_target_ref = get_base_target_ref_transform(H_base_gripper, H_camera_target, H_gripper_camera, display=False)

    total_sq_error_px = 0.0
    total_corners = 0
    H_camera_gripper = inv_SE3(H_gripper_camera)
    for i, (H_b_g, corners) in enumerate(zip(H_base_gripper, corners_list)):
        H_g_b = inv_SE3(H_b_g)
        pred_H_c_t = H_camera_gripper @ H_g_b @ H_base_target_ref
        
        imgpts = predict_corners(pred_H_c_t, inner_corners, square_size_m, K, D)  # (N,1,2)

        d = imgpts - corners  # (N,1,2)
        total_sq_error_px += float(np.sum(d ** 2))
        total_corners += corners.shape[0]
    rmse_px = float(np.sqrt(total_sq_error_px / total_corners))

    return rmse_px

def display_quality(H_base_gripper_list, H_camera_target_list, H_gripper_camera, img_path_arr, corners_arr, inner_corners, square_size_m, K, D, name=""):
    H_base_target_list = [H_base_gripper @ H_gripper_camera @ H_camera_target for H_base_gripper, H_camera_target in zip(H_base_gripper_list, H_camera_target_list)]
    R_list = [T[:3, :3] for T in H_base_target_list]
    t_stack = np.stack([T[:3, 3] for T in H_base_target_list], axis=0)
    visualize_r_t(R_list, t_stack, axis_len=0.005, title=f"Base->Target Poses {name}")
    R_ref = SciRot.from_matrix(np.stack([H[:3,:3] for H in H_base_target_list])).mean().as_matrix()
    t_ref = np.mean([H[:3,3] for H in H_base_target_list], axis=0).reshape(3,1)
    H_base_target_ref = stack_H(R_ref, t_ref)
    pred_H_camera_target_list = [inv_SE3(H_gripper_camera) @ inv_SE3(H_b_g) @ H_base_target_ref for H_b_g in H_base_gripper_list]
    pred_corners = [predict_corners(H_c_t, inner_corners, square_size_m, K, D) for H_c_t in pred_H_camera_target_list]
    visualize_corners(img_path_arr, corners_arr, pred_corners, title=f"{name} Calibration")

def handineye_residuals(x, H_base_gripper_list, corner_list, K, D, inner_corners, square_size_m):
    R_gripper_camera = cv2.Rodrigues(x[0:3])[0]
    t_gripper_camera = x[3:6].reshape(3,1)
    H_gripper_camera = stack_H(R_gripper_camera, t_gripper_camera)
    H_camera_gripper = inv_SE3(H_gripper_camera)


    R_base_target_ref = cv2.Rodrigues(x[6:9])[0]
    t_base_target_ref = x[9:12].reshape(3,1)
    H_base_target_ref = stack_H(R_base_target_ref, t_base_target_ref)

    H_gripper_base_list = [inv_SE3(H_b_g) for H_b_g in H_base_gripper_list]
    residuals = []
    for H_g_b, corners in zip(H_gripper_base_list, corner_list):
        pred_H_c_t = H_camera_gripper @ H_g_b @ H_base_target_ref

        imgpts = predict_corners(pred_H_c_t, inner_corners, square_size_m, K, D)  # (N,1,2)

        d = imgpts - corners  # (N,1,2)
        residuals.append(d.flatten())
    return np.concatenate(residuals)

def nonlinear_handeye_optimization(initial_H_gripper_camera, H_base_gripper_list, H_camera_target_list, corner_list, K, D, inner_corners, square_size_m):
    iterations = 10
    H_gripper_camera_opt = initial_H_gripper_camera.copy()
    H_base_target_list = [H_b_g @ H_gripper_camera_opt @ H_c_t for H_b_g, H_c_t in zip(H_base_gripper_list, H_camera_target_list)]
    R_ref = SciRot.from_matrix(np.stack([H[:3,:3] for H in H_base_target_list])).mean().as_matrix()
    t_ref = np.mean([H[:3,3] for H in H_base_target_list], axis=0).reshape(3,1)
    H_base_target_ref = stack_H(R_ref, t_ref)
    for _ in range(iterations):

        R_base_target_ref = H_base_target_ref[:3, :3]
        t_base_target_ref = H_base_target_ref[:3, 3].reshape(3,1)
        rvec_base_target_ref = cv2.Rodrigues(R_base_target_ref)[0].flatten()


        R_gripper_camera_opt = H_gripper_camera_opt[:3, :3]
        t_gripper_camera_opt = H_gripper_camera_opt[:3, 3].reshape(3,1)
        rvec_gripper_camera_opt = cv2.Rodrigues(R_gripper_camera_opt)[0].flatten()
        x0 = np.concatenate([rvec_gripper_camera_opt, t_gripper_camera_opt.flatten(), rvec_base_target_ref, t_base_target_ref.flatten()])

        result = opt.least_squares(
            handineye_residuals,
            x0,
            args=(H_base_gripper_list, corner_list, K, D, inner_corners, square_size_m),
            method="trf",
            loss="huber",           # or "soft_l1"
            f_scale=1.0,            # ≈ expected pixel noise (start with 1–2 px)
            x_scale="jac",          # auto parameter scaling
            xtol=1e-10, ftol=1e-10, gtol=1e-10,
            max_nfev=5000, verbose=0
        )
        x_opt = result.x
        R_opt = cv2.Rodrigues(x_opt[0:3])[0]
        t_opt = x_opt[3:6].reshape(3,1)
        H_gripper_camera_opt = stack_H(R_opt, t_opt)

        R_base_target_ref = cv2.Rodrigues(x_opt[6:9])[0]
        t_base_target_ref = x_opt[9:12].reshape(3,1)
        H_base_target_ref = stack_H(R_base_target_ref, t_base_target_ref)

    return H_gripper_camera_opt



def calibrate_handineye(data_dir, intrinsics_path, extrinsics_path, inner_corners, square_size_m, img_dir_path, display=True):
    # Load intrinsics
    K, D, distortion_model, width, height, R_rect, P = load_intrinsics_npz(intrinsics_path)
    print("[INFO] Intrinsics loaded:")
    print("K=\n", K)
    print("D=", D)
    print("R_rect=\n", R_rect)
    print("P=\n", P)
    print("distortion_model=", distortion_model)
    print(f"image size: {width} x {height}")

    print("beging loading data...")
    R_base_gripper_list, t_base_gripper_list, R_cam_target_list, t_cam_target_list, corners_arr, img_path_arr = load_data(data_dir, K, D, inner_corners, square_size_m, img_dir_path)
    print(f"[INFO] Loaded {len(R_base_gripper_list)} samples for hand-eye calibration.")
    best_H = np.eye(4)
    best_error = float('inf')
    ransac_iters = 200
    sample_n = 5
    for i in range(ransac_iters):

        sample_idxs = random.sample(range(len(R_base_gripper_list)), sample_n)
        R_base_gripper_sample = [R_base_gripper_list[i] for i in sample_idxs]
        t_base_gripper_sample = [t_base_gripper_list[i] for i in sample_idxs]
        R_cam_target_sample = [R_cam_target_list[i] for i in sample_idxs]
        t_cam_target_sample = [t_cam_target_list[i] for i in sample_idxs]

        R_gripper_cam, t_gripper_cam = cv2.calibrateHandEye(
            R_base_gripper_sample, t_base_gripper_sample,  # lists of absolutes are expected here
            R_cam_target_sample,  t_cam_target_sample,
            method= cv2.CALIB_HAND_EYE_TSAI
        )

        error_px = get_error(
            R_base_gripper_list, t_base_gripper_list,
            R_cam_target_list,  t_cam_target_list,
            R_gripper_cam, t_gripper_cam,
            corners_arr, K, D, inner_corners, square_size_m
        )
        print(f"{i}: {error_px=:.3f} px RMS")
        print("\n\n")

        if error_px < best_error:
            H_gripper_cam = stack_H(R_gripper_cam, t_gripper_cam)
            best_error = error_px
            best_H = H_gripper_cam.copy()

    print(f"[RESULT] Best reprojection error: {best_error:.3f} px RMS")
    print("Best H_gripper_camera:\n", best_H)


    H_base_gripper_list = [stack_H(R, t) for R, t in zip(R_base_gripper_list, t_base_gripper_list)]
    H_camera_target_list = [stack_H(R, t) for R, t in zip(R_cam_target_list, t_cam_target_list)]
    if display:
        H_gripper_camera = best_H
        display_quality(H_base_gripper_list, H_camera_target_list, H_gripper_camera, img_path_arr, corners_arr, inner_corners, square_size_m, K, D, name="TSAI Calibration")
        

    # # --- Nonlinear optimization ---
    # print("[INFO] Starting nonlinear optimization...")
    # best_H = nonlinear_handeye_optimization(best_H, H_base_gripper_list, H_camera_target_list, corners_arr, K, D, inner_corners, square_size_m)
    # H_gripper_camera = best_H
    # best_error = get_error(
    #     R_base_gripper_list, t_base_gripper_list,
    #     R_cam_target_list,  t_cam_target_list,
    #     best_H[:3, :3], best_H[:3, 3].reshape(3,1),
    #     corners_arr, K, D, inner_corners, square_size_m
    # )
    # print(f"[RESULT] After nonlinear optimization, reprojection error: {best_error:.3f} px RMS")
    # if display:
    #     H_gripper_camera = best_H
    #     display_quality(H_base_gripper_list, H_camera_target_list, H_gripper_camera, img_path_arr, corners_arr, inner_corners, square_size_m, K, D, name="OPT Least Squares Calibration")


    H_gripper_cameraopt = best_H.copy()
    H_camerabase_cameraoptical = np.load(extrinsics_path, allow_pickle=True)["H_cameraoptical_camerabase"]

    H_gripper_camerabase = H_gripper_cameraopt @ inv_SE3(H_camerabase_cameraoptical)
    

    R_final = H_gripper_camerabase[:3, :3]
    t_final = H_gripper_camerabase[:3,  3]


    # Output values
    x, y, z = t_final.flatten().tolist()
    qx, qy, qz, qw = SciRot.from_matrix(R_final).as_quat()  # xyzw
    print(f"{x=:.6f}, {y=:.6f}, {z=:.6f}")
    print(f"{qx=}, {qy=}, {qz=}, {qw=}")

    print()
    print()
    print(f"'{x}', '{y}', '{z}',")
    print(f"'{qx}', '{qy}', '{qz}', '{qw}',")

    return H_gripper_cameraopt, best_error

def main():
    INNER_CORNERS = (10, 7)      # (cols, rows)
    SQUARE_SIZE_M = 0.020         # 2cm
    file_location = os.path.dirname(os.path.abspath(__file__))
    print(f"File location: {file_location}")
    UR = False
    experiment_str = "handineye_calibration"
    if UR:
        experiment_str += "_ur"
    else:
        experiment_str += "_h12"
    DATA_DIR = os.path.join(file_location, "data", experiment_str, "npzs")
    assert os.path.exists(DATA_DIR), f"Data dir not found: {DATA_DIR}"
    INTRINSICS_PATH = os.path.join(file_location, "data", experiment_str, "intrinsics_0.npz")
    assert os.path.exists(INTRINSICS_PATH), f"Intrinsics file not found: {INTRINSICS_PATH}"
    EXTRINSICS_PATH = os.path.join(file_location, "data", experiment_str, "extrinsics_0.npz")
    assert os.path.exists(EXTRINSICS_PATH), f"Extrinsics file not found: {EXTRINSICS_PATH}"
    IMG_DIR_PATH = os.path.join(file_location, "data", experiment_str, "raw")
    assert os.path.exists(IMG_DIR_PATH), f"Image dir not found: {IMG_DIR_PATH}"
    H_gripper_cameraopt, best_error = calibrate_handineye(DATA_DIR, INTRINSICS_PATH, EXTRINSICS_PATH, INNER_CORNERS, SQUARE_SIZE_M, IMG_DIR_PATH)
    return H_gripper_cameraopt, best_error
if __name__ == "__main__":
    main()
