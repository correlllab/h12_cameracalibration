import os
import glob
from typing import List, Tuple
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as SciRot
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import glob
from utils import load_intrinsics_npz, H_cam_target_from_corners, inv_SE3
from scipy.optimize import least_squares
from handineye_calibration_computation import main as calibrate_handineye
from handtoeye_calibration_computation import main as calibrate_handtoeye

def compute_reprojection_error(H_base_ee_list, corners_cam0_list, corners_cam1_list,
        H_cam0_target_list, H_cam1_target_list,
        cam0_K, cam0_D,
        cam1_K, cam1_D,
        H_ee_cam0optical,
        H_base_cam1optical,
        inner_corners, square_size_m
        ) -> float:
    target_points = np.zeros((inner_corners[0]*inner_corners[1], 3), np.float32)
    target_points[:, :2] = np.mgrid[0:inner_corners[0], 0:inner_corners[1]].T.reshape(-1, 2)
    target_points *= float(square_size_m)

    cam0_error_accum = 0.0
    cam1_error_accum = 0.0
    cam0_N_points = 0
    cam1_N_points = 0
    
    N = len(H_base_ee_list)
    H_cam0optical_ee = inv_SE3(H_ee_cam0optical)
    H_cam1optical_base = inv_SE3(H_base_cam1optical)
    for i in range(N):
        # cam0 <- target (pred)
        H_ee_base = inv_SE3(H_base_ee_list[i])
        H_cam0_target_pred = H_cam0optical_ee @ H_ee_base @ H_base_cam1optical @ H_cam1_target_list[i]
        H_cam1_target_pred = H_cam1optical_base @ H_base_ee_list[i] @ H_ee_cam0optical @ H_cam0_target_list[i]

        # sanity: board in front of cam0
        if (H_cam0_target_pred @ np.array([0,0,0,1.0]))[2] <= 0:
            raise ValueError("Target is behind cam0!")
        if (H_cam1_target_pred @ np.array([0,0,0,1.0]))[2] <= 0:
            raise ValueError("Target is behind cam1!")

        # ----- cam0 reprojection -----
        R_c0_t = H_cam0_target_pred[:3,:3].astype(np.float64)
        t_c0_t = H_cam0_target_pred[:3, 3].reshape(3,1).astype(np.float64)
        c0_rvec, _ = cv2.Rodrigues(R_c0_t)
        c0_pred_corners, _ = cv2.projectPoints(target_points.astype(np.float64), c0_rvec, t_c0_t,
                                    cam0_K.astype(np.float64),
                                    None if cam0_D is None else cam0_D.astype(np.float64))
        c0_pred_corners = c0_pred_corners.reshape(-1, 2)

        c0_true_corners = corners_cam0_list[i]
        if c0_true_corners.ndim == 3:  # (N,1,2)
            c0_true_corners = c0_true_corners.reshape(-1,2)

        c0_diff = c0_pred_corners - c0_true_corners.astype(np.float64)  # (N,2)
        cam0_error_accum += float(np.sum(c0_diff * c0_diff))
        cam0_N_points    += c0_true_corners.shape[0]


        R_c1_t = H_cam1_target_pred[:3,:3].astype(np.float64)
        t_c1_t = H_cam1_target_pred[:3, 3].reshape(3,1).astype(np.float64)
        c1_rvec, _ = cv2.Rodrigues(R_c1_t)
        c1_pred_corners, _ = cv2.projectPoints(target_points.astype(np.float64), c1_rvec, t_c1_t,
                                     cam1_K.astype(np.float64),
                                     None if cam1_D is None else cam1_D.astype(np.float64))
        c1_pred_corners = c1_pred_corners.reshape(-1, 2)

        c1_true_corners = corners_cam1_list[i]
        if c1_true_corners.ndim == 3:
            c1_true_corners = c1_true_corners.reshape(-1,2)

        diff1 = c1_pred_corners - c1_true_corners.astype(np.float64)
        cam1_error_accum += float(np.sum(diff1 * diff1))
        cam1_N_points    += c1_true_corners.shape[0]

    cam0_rmse = np.sqrt(cam0_error_accum / cam0_N_points) if cam0_N_points else float('nan')
    cam1_rmse = np.sqrt(cam1_error_accum / cam1_N_points) if cam1_N_points else float('nan')

    print(f"Cam0 RMSE: {cam0_rmse}")
    print(f"Cam1 RMSE: {cam1_rmse}")
    return cam0_rmse + cam1_rmse




def bundle_adjustment(H_base_ee_list, corners_cam0_list, corners_cam1_list,
                      H_target_cam0_list, H_target_cam1_list,
                      cam0_K, cam0_D,
                      cam1_K, cam1_D,
                      initial_ee_cam0optical,
                      initial_base_cam1optical,
                      inner_corners, square_size_m
                      ) -> Tuple[np.ndarray, np.ndarray]:
    return


def main():
    INNER_CORNERS = (10, 7)      # (cols, rows)
    SQUARE_SIZE_M = 0.020         # 2cm
    

    file_location = os.path.dirname(os.path.abspath(__file__))
    print(f"File location: {file_location}")
    DATA_DIR = os.path.join(file_location, "data", "bundle_data")
    NPZ_DIR = os.path.join(DATA_DIR, "npzs")
    assert os.path.exists(DATA_DIR), f"Data dir not found: {DATA_DIR}"
    assert os.path.exists(NPZ_DIR), f"NPZ dir not found: {NPZ_DIR}"
    CAM0_INTRINSICS_PATH = os.path.join(DATA_DIR, "intrinsics_0.npz")
    CAM1_INTRINSICS_PATH = os.path.join(DATA_DIR, "intrinsics_1.npz")
    CAM0_EXTRINSICS_PATH = os.path.join(DATA_DIR, "extrinsics_0.npz")
    CAM1_EXTRINSICS_PATH = os.path.join(DATA_DIR, "extrinsics_1.npz")
    assert os.path.exists(CAM0_INTRINSICS_PATH), f"Cam0 Intrinsics file not found: {CAM0_INTRINSICS_PATH}"
    assert os.path.exists(CAM1_INTRINSICS_PATH), f"Cam1 Intrinsics file not found: {CAM1_INTRINSICS_PATH}"
    assert os.path.exists(CAM0_EXTRINSICS_PATH), f"Cam0 Extrinsics file not found: {CAM0_EXTRINSICS_PATH}"
    assert os.path.exists(CAM1_EXTRINSICS_PATH), f"Cam1 Extrinsics file not found: {CAM1_EXTRINSICS_PATH}"

    cam0_K, cam0_D, _, _, _, _, _ = load_intrinsics_npz(CAM0_INTRINSICS_PATH)
    cam1_K, cam1_D, _, _, _, _, _ = load_intrinsics_npz(CAM1_INTRINSICS_PATH)

    cam0_npz = glob.glob(os.path.join(NPZ_DIR, "*_cam_0.npz"))
    cam1_npz = glob.glob(os.path.join(NPZ_DIR, "*_cam_1.npz"))
    print(f"Found {len(cam0_npz)} cam0 npzs and {len(cam1_npz)} cam1 npzs")
    assert len(cam0_npz) == len(cam1_npz), "Mismatched number of cam0 and cam1 npzs"

    corners_cam0_list = []
    corners_cam1_list = []
    H_cam0_target_list = []
    H_cam1_target_list = []
    H_base_ee_list = []
    for cam0_npz_path in cam0_npz:
        cam1_npz_path = cam0_npz_path.replace("_cam_0.npz", "_cam_1.npz")
        assert os.path.exists(cam1_npz_path), f"Cam1 npz not found for {cam0_npz_path}"

        cam0_data = np.load(cam0_npz_path, allow_pickle=True)
        cam1_data = np.load(cam1_npz_path, allow_pickle=True)

        corners_cam0 = cam0_data["corners"]
        H_cam0_target, error = H_cam_target_from_corners(corners_cam0, cam0_K, cam0_D, INNER_CORNERS, SQUARE_SIZE_M)
        print(f"Cam0 reprojection error: {error:.3f} px RMS")
        corners_cam1 = cam1_data["corners"]
        H_cam1_target, error = H_cam_target_from_corners(corners_cam1, cam1_K, cam1_D, INNER_CORNERS ,SQUARE_SIZE_M)
        print(f"Cam1 reprojection error: {error:.3f} px RMS")
        

        H_base_gripper_cam0 = cam0_data["T_base_ee"]
        H_base_gripper_cam1 = cam1_data["T_base_ee"]
        assert np.allclose(H_base_gripper_cam0, H_base_gripper_cam1), "Mismatched H_base_gripper between cam0 and cam1"

        H_cam0_target_list.append(H_cam0_target)
        H_cam1_target_list.append(H_cam1_target)
        corners_cam0_list.append(corners_cam0)
        corners_cam1_list.append(corners_cam1)
        H_base_ee_list.append(H_base_gripper_cam0)

    N = len(corners_cam0_list)
    assert len(corners_cam1_list) == N
    assert len(H_cam0_target_list) == N
    assert len(H_cam1_target_list) == N
    assert len(H_base_ee_list) == N


    initial_ee_cam0optical, _ = calibrate_handineye()
    initial_base_cam1optical, _ = calibrate_handtoeye()
    print("Initial Cam0 Transform:\n", initial_ee_cam0optical)
    print("Initial Cam1 Transform:\n", initial_base_cam1optical)

    initial_error = compute_reprojection_error(
        H_base_ee_list, corners_cam0_list, corners_cam1_list,
        H_cam0_target_list, H_cam1_target_list,
        cam0_K, cam0_D,
        cam1_K, cam1_D,
        initial_ee_cam0optical,
        initial_base_cam1optical,
        INNER_CORNERS, SQUARE_SIZE_M
    )
    print(f"Initial reprojection error: {initial_error:.3f} px RMS")


    # final_ee_cam0optical, final_base_cam1optical = bundle_adjustment(
    #     H_base_ee_list, corners_cam0_list, corners_cam1_list,
    #     H_target_cam0_list, H_target_cam1_list,
    #     cam0_K, cam0_D,
    #     cam1_K, cam1_D,
    #     initial_ee_cam0optical,
    #     initial_base_cam1optical,
    #     INNER_CORNERS, SQUARE_SIZE_M
    # )



    # H_cam0base_cam0optical = np.load(CAM0_EXTRINSICS_PATH, allow_pickle=True)["T_camerabase_cameraoptical"]
    # H_cam1base_cam1optical = np.load(CAM1_EXTRINSICS_PATH, allow_pickle=True)["T_camerabase_cameraoptical"]
    


if __name__ == "__main__":
    main()