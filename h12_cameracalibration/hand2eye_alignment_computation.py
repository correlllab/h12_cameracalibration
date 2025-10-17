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
DATA_DIR = "/ros2_ws/src/h12_cameracalibration/h12_cameracalibration/data/hand2eye_alignment"
assert os.path.exists(DATA_DIR), f"Data dir not found: {DATA_DIR}"
npz_save_dir = os.path.join(DATA_DIR, "npzs")
KNOWN_INTRINSICS_PATH = os.path.join(DATA_DIR, "known_intrinsics.npz")
UNKNOWN_INTRINSICS_PATH = os.path.join(DATA_DIR, "unknown_intrinsics.npz")
UNKNOWN_EXTRINSICS_PATH = os.path.join(DATA_DIR, "unknown_extrinsics.npz")

INNER_CORNERS = (10, 7)      # (cols, rows)
SQUARE_SIZE_M = 0.020         # 1cm


from handineye_calibration_computation import load_intrinsics_npz, target2cam_from_corners
def inv_SE3(T):
    """Inverse of a rigid homogeneous transform (rotation+translation)."""
    R = T[:3,:3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Rt = R.T
    Ti[:3,:3] = Rt
    Ti[:3, 3] = -Rt @ t
    return Ti

def main(known_intrinsic_path, unknown_intrinsic_path, unknown_extrinsics_path):
    print("\n\n\nComputing hand-to-eye calibration...")
    known_K, known_D, _, _, _, _, _ = load_intrinsics_npz(known_intrinsic_path)
    unknown_K, unknown_D, _, _, _, _, _ = load_intrinsics_npz(unknown_intrinsic_path)
    unknown_cam2optical = np.load(unknown_extrinsics_path, allow_pickle=True)["cam2optical"]
    npz_files = sorted(glob.glob(os.path.join(npz_save_dir, "*.npz")))

    known_corners_list = []
    unknown_corners_list = []
    T_target2known_list = []
    T_target2unknown_list = []
    T_known2base_list = []
    left_out = 0
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        kc = data["known_corners"]
        uc = data["unknown_corners"]
        T_k2b = data["T_known2base"]

        t2k, k_reproj_error = target2cam_from_corners(kc, known_K, known_D, INNER_CORNERS, SQUARE_SIZE_M)
        t2u, u_reproj_error = target2cam_from_corners(uc, unknown_K, unknown_D, INNER_CORNERS, SQUARE_SIZE_M)
        print(f"{npz_file} {k_reproj_error=}, {u_reproj_error=}")
        if k_reproj_error > 1 or u_reproj_error > 1:
            print(f"Skipping {npz_file} due to high reproj error")
            left_out += 1
            continue
        known_corners_list.append(kc)
        unknown_corners_list.append(uc)
        T_target2known_list.append(t2k)
        T_target2unknown_list.append(t2u)
        T_known2base_list.append(T_k2b)

    print(f"Left out {left_out} frames due to high reprojection error")
    N = len(known_corners_list)
    assert len(unknown_corners_list) == N
    assert len(T_target2known_list) == N
    assert len(T_target2unknown_list) == N
    assert len(T_known2base_list) == N
    if N < 2:
        print("Not enough valid frames to compute hand-to-eye calibration")
        return None
    T_base2unknown_list = []
    reproj_errors = []


    target_points = np.zeros((INNER_CORNERS[0]*INNER_CORNERS[1], 3), np.float32)
    target_points[:, :2] = np.mgrid[0:INNER_CORNERS[0], 0:INNER_CORNERS[1]].T.reshape(-1, 2)
    target_points *= SQUARE_SIZE_M
    # print(f"[DEBUG] objp:\n{objp}")


    for i in range(N):
        T_t2k = T_target2known_list[i]
        T_t2u = T_target2unknown_list[i]
        T_k2b = T_known2base_list[i]

        T_b2u = inv_SE3(T_k2b) @ inv_SE3(T_t2k) @ T_t2u
        T_base2unknown_list.append(T_b2u)


        pred_i = T_target2known_list[i] @ T_known2base_list[i] @ T_b2u
        rvec_i, _ = cv2.Rodrigues(pred_i[:3,:3].astype(np.float64))
        tvec_i = pred_i[:3,3].astype(np.float64).reshape(3,1)
        proj_i, _ = cv2.projectPoints(
            target_points.astype(np.float32).reshape(-1,1,3),
            rvec_i, tvec_i, unknown_K.astype(np.float64), unknown_D.astype(np.float64)
        )
        rmse_i = float(np.sqrt(np.mean(np.sum(
            (proj_i.reshape(-1,2) - unknown_corners_list[i].reshape(-1,2))**2, axis=1))))
        print(f"[self-check] frame {i} rmse = {rmse_i:.3f}px")

        error_acc = 0
        for j in range(N):
            if i == j:
                continue
            # pred_t2u = inv_SE3(T_target2known_list[j]) @ inv_SE3(T_known2base_list[j]) @ T_b2u
            pred_t2u = T_target2known_list[j] @ T_known2base_list[j] @ T_b2u
            R = pred_t2u[:3, :3]
            t = pred_t2u[:3, 3]
            rvec, _ = cv2.Rodrigues(R.astype(np.float64))
            tvec = t.reshape(3,1).astype(np.float64)
            

            pred_corners, _ = cv2.projectPoints(
                target_points.astype(np.float32).reshape(-1,1,3),
                rvec, tvec, unknown_K.astype(np.float64), unknown_D.astype(np.float64)
            )
            
            true_corners = unknown_corners_list[j]

            diff = (pred_corners.reshape(-1, 2) - true_corners.reshape(-1, 2)).astype(np.float64)

            # Per-corner radial errors (L2 per point)
            per_corner = np.linalg.norm(diff, axis=1)

            # RMSE (radial, in pixels)
            rmse = float(np.sqrt(np.mean(per_corner**2)))

            error_acc += rmse
        reproj_errors.append(error_acc / (N - 1))


    for T_b2u in T_base2unknown_list:
        print(f"T_base2unknown:\n{T_b2u}\n")

    best_idx = np.argmin(reproj_errors)
    best_T = T_base2unknown_list[best_idx]
    # best_T = best_T @ unknown_cam2optical
    print(f"Best idx: {best_idx}, reproj error: {reproj_errors[best_idx]}")
    print(f"Best T_base2unknown:\n{best_T}")
    return best_T

if __name__ == "__main__":
    best_T = main(KNOWN_INTRINSICS_PATH, UNKNOWN_INTRINSICS_PATH, UNKNOWN_EXTRINSICS_PATH)


    R_final = best_T[:3, :3]
    t_final = best_T[:3,  3]


    # Output values

    x, y, z = t_final.flatten().tolist()
    qx, qy, qz, qw = SciRot.from_matrix(R_final).as_quat()  # xyzw
    print(f"{x=:.6f}, {y=:.6f}, {z=:.6f}")
    print(f"{qx=}, {qy=}, {qz=}, {qw=}")

    print()
    print()
    print(f"'{x}', '{y}', '{z}',")
    print(f"'{qx}', '{qy}', '{qz}', '{qw}',")

    exit(0)