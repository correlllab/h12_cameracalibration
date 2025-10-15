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
EXTRINSICS_PATH = "/ros2_ws/src/h12_cameracalibration/h12_cameracalibration/data/handtoeye_calibration/extrinsics.npz"
assert os.path.exists(EXTRINSICS_PATH), f"Extrinsics file not found: {EXTRINSICS_PATH}"

INNER_CORNERS = (7, 10)      # (cols, rows)
SQUARE_SIZE_M = 0.010         # 1cm

from handineye_calibration_computation import visualize_r_t, stack_T, load_intrinsics_npz, target2cam_from_corners, HAND_EYE_METHODS
def inv_SE3(T):
    """Inverse of a rigid homogeneous transform (rotation+translation)."""
    R = T[:3,:3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Rt = R.T
    Ti[:3,:3] = Rt
    Ti[:3, 3] = -Rt @ t
    return Ti

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
    rs2optical = np.load(EXTRINSICS_PATH, allow_pickle=True)["cam2optical"]
    print(f"[INFO] Found {len(npz_files)} NPZ files in {DATA_DIR}")
    R_base2gripper = []
    t_base2gripper = []
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
        R_gripper2base.append(T_gripper2base[:3, :3])
        t_gripper2base.append(T_gripper2base[:3, 3].reshape(3,1))
         # Invert to get base to gripper


        T_base2gripper = inv_SE3(T_gripper2base)
        R_base2gripper.append(T_base2gripper[:3, :3])
        t_base2gripper.append(T_base2gripper[:3, 3].reshape(3,1))
        
    print(f"[INFO] Rejected {rejected} samples due to high reprojection error.")
    assert len(R_base2gripper) == len(t_base2gripper) == len(R_target2cam) == len(t_target2cam)
    print(f"[INFO] Loaded {len(R_base2gripper)} samples")
    print(f"[INFO] Example shape: {R_base2gripper[0].shape=}, {t_base2gripper[0].shape=} {R_target2cam[0].shape=}, {t_target2cam[0].shape=}")

    # visualize_r_t(R_base2gripper, t_base2gripper, title="Base to Gripper Poses")
    # visualize_r_t(R_gripper2base, t_gripper2base, title="Gripper to Base Poses")
    # visualize_r_t(R_target2cam, t_target2cam, title="Target to Camera Poses")
    # plt.show()

    best_error = float('inf')
    iters = 20
    sample_n = 7
    solutions = []
    for i in range(iters):

        sample_idxs = random.sample(range(len(R_base2gripper)), sample_n)
        R_base2gripper_sample = [R_base2gripper[i] for i in sample_idxs]
        t_base2gripper_sample = [t_base2gripper[i] for i in sample_idxs]
        R_target2cam_sample = [R_target2cam[i] for i in sample_idxs]
        t_target2cam_sample = [t_target2cam[i] for i in sample_idxs]

        R_base2cam, t_base2cam = cv2.calibrateHandEye(
            R_base2gripper_sample, t_base2gripper_sample,
            R_target2cam_sample,  t_target2cam_sample,
            method=HAND_EYE_METHODS["TSAI"]
        )
        T_base2cam = np.eye(4)
        T_base2cam[:3, :3] = R_base2cam
        T_base2cam[:3, 3] = t_base2cam.flatten()

        solutions.append(T_base2cam)

        # print(f"{T_base2cam}")
    print(f"[RESULT] Best reprojection error: {best_error:.3f} mm RMS")

    R_base2cam_all = [T[:3, :3] for T in solutions]
    t_base2cam_all = [T[:3,  3].reshape(3,1) for T in solutions]
    visualize_r_t(R_base2cam_all, t_base2cam_all, title="Base to Camera Poses (All Solutions)")
    plt.show()

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

if __name__ == "__main__":
    main()
