import os
import glob
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as SciRot
import random
import matplotlib.pyplot as plt
from utils import stack_H, visualize_r_t, load_intrinsics_npz, load_data, inv_SE3


def get_error(H_cam_base, H_cam_target_list, H_base_gripper_list):
    
    return total_error / n



def calibrate_handtoeye(data_dir, intrinsics_path, extrinsics_path, inner_corners, square_size_m, img_dir_path, display = False):
    # Load intrinsics
    K, D, distortion_model, width, height, R_rect, P = load_intrinsics_npz(intrinsics_path)
    print("[INFO] Intrinsics loaded:")
    print("K=\n", K)
    print("D=", D)
    print("distortion_model=", distortion_model)
    print(f"image size: {width} x {height}")
    R_base_gripper_list, t_base_gripper_list, R_cam_target_list, t_cam_target_list, corners_arr, img_path_arr = load_data(data_dir, K, D, inner_corners, square_size_m, img_dir_path)

    # visualize_r_t(R_target2cam, t_target2cam)
    R_gripper_base_list = []
    t_gripper_base_list = []
    for R_base_gripper, t_base_gripper in zip(R_base_gripper_list, t_base_gripper_list):
        H_base_gripper = stack_H(R_base_gripper, t_base_gripper)
        H_gripper_base = inv_SE3(H_base_gripper)
        R_gripper_base_list.append(H_gripper_base[:3, :3])
        t_gripper_base_list.append(H_gripper_base[:3, 3].reshape(3,1))

    R_cam_base, t_cam_base = cv2.calibrateHandEye(
        R_gripper_base_list, t_gripper_base_list,
        R_cam_target_list,  t_cam_target_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    H_cam_base = stack_H(R_cam_base, t_cam_base)

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