import os
import glob
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as SciRot
import random
import matplotlib.pyplot as plt
from utils import stack_H, visualize_r_t, load_intrinsics_npz, load_data, inv_SE3, predict_corners, visualize_corners


def get_error(H_cam_base, H_cam_target_list, H_base_gripper_list,
              corners_arr, K, D, inner_corners, square_size_m, img_arr=None):
    if not corners_arr:
        return np.nan

    # Inverses
    H_gripper_base_list = [inv_SE3(H_bg) for H_bg in H_base_gripper_list]
    H_target_cam_list   = [inv_SE3(H_ct) for H_ct in H_cam_target_list]

    # Target←gripper for each capture:  T←G = (T←C)(C←B)(B←G)
    H_target_gripper_list = [
        H_tc @ H_cam_base @ H_bg
        for H_tc, H_bg in zip(H_target_cam_list, H_base_gripper_list)
    ]

    # Reference T←G via rotation mean + translation mean
    R_ref = SciRot.from_matrix(np.stack([H[:3, :3] for H in H_target_gripper_list])).mean().as_matrix()
    t_ref = np.mean([H[:3, 3] for H in H_target_gripper_list], axis=0).reshape(3, 1)
    H_target_gripper_ref = stack_H(R_ref, t_ref)

    # Predict camera←target for each capture:  C←T = (C←B)(B←G)(G←T)
    H_cam_target_pred = [
        H_cam_base @ H_bg @ inv_SE3(H_target_gripper_ref)
        for H_bg in H_base_gripper_list
    ]

    # Project corners
    pred_corners = [
        predict_corners(H_ct, inner_corners, square_size_m, K, D)
        for H_ct in H_cam_target_pred
    ]

    # Per-corner RMSE (scalar)
    dist_sq_sum = 0.0
    corner_count = 0
    for corners, pred in zip(corners_arr, pred_corners):
        d = corners - pred              # (n, 2)
        dist_sq_sum += float(np.sum(d**2))
        corner_count += d.shape[0]

    if corner_count == 0:
        return np.nan

    rmse = np.sqrt(dist_sq_sum / corner_count)

    print(f"Total reprojection RMSE over {corner_count} corners: {rmse:.6f} px")
    if img_arr is not None:
        visualize_corners(img_arr, corners_arr, pred_corners, title="Reprojection Error Visualization")
    return rmse



def calibrate_handtoeye(data_dir, intrinsics_path, extrinsics_path, inner_corners, square_size_m, img_dir_path, display = False):
    # Load intrinsics
    K, D, distortion_model, width, height, R_rect, P = load_intrinsics_npz(intrinsics_path)
    print("[INFO] Intrinsics loaded:")
    print("K=\n", K)
    print("D=", D)
    print("distortion_model=", distortion_model)
    print(f"image size: {width} x {height}")
    R_base_gripper_list, t_base_gripper_list, R_cam_target_list, t_cam_target_list, corners_arr, img_path_arr = load_data(data_dir, K, D, inner_corners, square_size_m, img_dir_path)

    H_base_gripper_list = [stack_H(R, t) for R, t in zip(R_base_gripper_list, t_base_gripper_list)]
    H_cam_target_list = [stack_H(R, t) for R, t in zip(R_cam_target_list, t_cam_target_list)]
    # visualize_r_t(R_target2cam, t_target2cam)
    H_gripper_base_list = []
    R_gripper_base_list = []
    t_gripper_base_list = []
    for R_base_gripper, t_base_gripper in zip(R_base_gripper_list, t_base_gripper_list):
        H_base_gripper = stack_H(R_base_gripper, t_base_gripper)
        H_gripper_base = inv_SE3(H_base_gripper)
        H_gripper_base_list.append(H_gripper_base)
        R_gripper_base_list.append(H_gripper_base[:3, :3])
        t_gripper_base_list.append(H_gripper_base[:3, 3].reshape(3,1))

    ransac_iters = 200
    sample_n = 5
    best_error = float('inf')
    best_H = np.eye(4)
    for i in range(ransac_iters):
        sample_indices = random.sample(range(len(R_base_gripper_list)), sample_n)
        R_gripper_base_sample = [R_gripper_base_list[i] for i in sample_indices]
        t_gripper_base_sample = [t_gripper_base_list[i] for i in sample_indices]
        R_cam_target_sample = [R_cam_target_list[i] for i in sample_indices]
        t_cam_target_sample = [t_cam_target_list[i] for i in sample_indices]

        R_cam_base, t_cam_base = cv2.calibrateHandEye(
            R_gripper_base_sample, t_gripper_base_sample,
            R_cam_target_sample,  t_cam_target_sample,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        H_cam_base = stack_H(R_cam_base, t_cam_base)
        error = get_error(H_cam_base, H_cam_target_list, H_base_gripper_list, corners_arr, K, D, inner_corners, square_size_m, None)
        if i == 0 or error < best_error:
            best_error = error
            best_H = H_cam_base
    
        print(f"reprojection error: {error:.6f} px")
    print(f"Best reprojection error: {error:.6f} px")
    H_cam_base = best_H
    error = get_error(H_cam_base, H_cam_target_list, H_base_gripper_list, corners_arr, K, D, inner_corners, square_size_m, img_path_arr)
    if display:
        visualize_r_t([R_cam_base], [t_cam_base], title="Base to Camera Pose")

    H_camerabase_cameraoptical = np.load(extrinsics_path, allow_pickle=True)["T_camerabase_cameraoptical"]

    H_camopt_base = H_cam_base.copy()

    # H_cambase_base = H_camerabase_cameraoptical @ H_camopt_base #inv_SE3(H_camerabase_cameraoptical)
    H_cambase_base = H_camopt_base @ H_camerabase_cameraoptical 
    R_final = H_cambase_base[:3, :3]
    t_final = H_cambase_base[:3,  3]


    # Output values
    x, y, z = t_final.flatten().tolist()
    qx, qy, qz, qw = SciRot.from_matrix(R_final).as_quat()  # xyzw
    print(f"{x=:.6f}, {y=:.6f}, {z=:.6f}")
    print(f"{qx=}, {qy=}, {qz=}, {qw=}")

    print()
    print()
    print(f"'{x}', '{y}', '{z}',")
    print(f"'{qx}', '{qy}', '{qz}', '{qw}',")

    # plt.show()
    error = 0.0  # --- IGNORE ---
    return H_camopt_base, error

def main():
    INNER_CORNERS = (10, 7)      # (cols, rows)
    SQUARE_SIZE_M = 0.010         # 1cm
    file_location = os.path.dirname(os.path.abspath(__file__))
    print(f"File location: {file_location}")
    DATA_DIR = os.path.join(file_location, "data", "handtoeye_calibration", "npzs")
    assert os.path.exists(DATA_DIR), f"Data dir not found: {DATA_DIR}"
    INTRINSICS_PATH = os.path.join(file_location, "data", "handtoeye_calibration", "intrinsics_0.npz")
    assert os.path.exists(INTRINSICS_PATH), f"Intrinsics file not found: {INTRINSICS_PATH}"
    EXTRINSICS_PATH = os.path.join(file_location, "data", "handtoeye_calibration", "extrinsics_0.npz")
    assert os.path.exists(EXTRINSICS_PATH), f"Extrinsics file not found: {EXTRINSICS_PATH}"
    IMG_DIR_PATH = os.path.join(file_location, "data", "handtoeye_calibration", "raw")
    assert os.path.exists(IMG_DIR_PATH), f"Image dir not found: {IMG_DIR_PATH}"
    return calibrate_handtoeye(DATA_DIR, INTRINSICS_PATH, EXTRINSICS_PATH, INNER_CORNERS, SQUARE_SIZE_M, IMG_DIR_PATH)

if __name__ == "__main__":
    main()