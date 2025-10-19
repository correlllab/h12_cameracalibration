import os
import glob
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as SciRot
import random
import matplotlib.pyplot as plt


from utils import stack_T, visualize_r_t, load_intrinsics_npz, load_data, inv_SE3


def calibrate_handtoeye(data_dir, intrinsics_path, extrinsics_path, inner_corners, square_size_m):
    # Load intrinsics
    K, D, distortion_model, width, height, R_rect, P = load_intrinsics_npz(intrinsics_path)
    rs2optical = np.load(extrinsics_path, allow_pickle=True)["T_camerabase_cameraoptical"]
    print("[INFO] Intrinsics loaded:")
    print("K=\n", K)
    print("D=", D)
    print("distortion_model=", distortion_model)
    print(f"image size: {width} x {height}")
    R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, corners_arr = load_data(data_dir, K, D, inner_corners, square_size_m)
    R_base2gripper = []
    t_base2gripper = []
    for R, t in zip(R_gripper2base, t_gripper2base):
        T_gripper2base = stack_T(R, t)
        T_base2gripper = inv_SE3(T_gripper2base)
        R_base2gripper.append(T_base2gripper[:3, :3])
        t_base2gripper.append(T_base2gripper[:3, 3].reshape(3,1))
       
    R_base2cam, t_base2cam = cv2.calibrateHandEye(
        R_base2gripper, t_base2gripper,
        R_target2cam,  t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    T_base2cam = np.eye(4)
    T_base2cam[:3, :3] = R_base2cam
    T_base2cam[:3, 3] = t_base2cam.flatten()

    visualize_r_t([R_base2cam], [t_base2cam], title="Base to Camera Pose")

    T_base2camOpt = T_base2cam.copy()

    T_base2cam = T_base2camOpt @ rs2optical
    

    R_final = T_base2cam[:3, :3]
    t_final = T_base2cam[:3,  3]


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
    return T_base2cam

if __name__ == "__main__":
    INNER_CORNERS = (10, 7)      # (cols, rows)
    SQUARE_SIZE_M = 0.020         # 2cm
    file_location = os.path.dirname(os.path.abspath(__file__))
    print(f"File location: {file_location}")
    DATA_DIR = os.path.join(file_location, "data", "handtoeye_calibration", "npzs")
    assert os.path.exists(DATA_DIR), f"Data dir not found: {DATA_DIR}"
    INTRINSICS_PATH = os.path.join(file_location, "data", "handtoeye_calibration", "intrinsics_0.npz")
    assert os.path.exists(INTRINSICS_PATH), f"Intrinsics file not found: {INTRINSICS_PATH}"
    EXTRINSICS_PATH = os.path.join(file_location, "data", "handtoeye_calibration", "extrinsics.npz")
    assert os.path.exists(EXTRINSICS_PATH), f"Extrinsics file not found: {EXTRINSICS_PATH}"
    calibrate_handtoeye(DATA_DIR, INTRINSICS_PATH, EXTRINSICS_PATH, INNER_CORNERS, SQUARE_SIZE_M)
