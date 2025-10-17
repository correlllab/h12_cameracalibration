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
from controller import ControllerNode
def inv_SE3(T):
    """Inverse of a rigid homogeneous transform (rotation+translation)."""
    R = T[:3,:3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Rt = R.T
    Ti[:3,:3] = Rt
    Ti[:3, 3] = -Rt @ t
    return Ti
from scipy.optimize import least_squares

def rvec_t_to_T(rvec, t):
    R, _ = cv2.Rodrigues(rvec.astype(np.float64).reshape(3,1))
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t.astype(np.float64).reshape(3)
    return T

def T_to_rvec_t(T):
    R = T[:3,:3].astype(np.float64)
    # Orthonormalize (safety)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0: R[:, -1] *= -1
    rvec, _ = cv2.Rodrigues(R)
    t = T[:3,3].astype(np.float64).reshape(3,1)
    return rvec.reshape(3), t.reshape(3)

def ba_refine_Tb2u(
    T_b2u_init,
    target_points,                # (N,3) float32/64 in meters
    unknown_K, unknown_D,         # intrinsics/distortion of UNKNOWN cam
    T_target2known_list,          # list of 4x4
    T_target2unknown_list,        # list of 4x4 (not directly used here, but kept for clarity)
    T_known2base_list,            # list of 4x4
    unknown_corners_list,         # list of (N,1,2)
    huber_scale=2.0,              # Huber f_scale (≈ pixels)
    verbose=2
):    
    T_t2b_list = [T_t2k @ T_k2b for T_t2k, T_k2b in zip(T_target2known_list, T_known2base_list)]

    objp = target_points.reshape(-1,1,3).astype(np.float32)
    K = unknown_K.astype(np.float64)
    D = unknown_D.astype(np.float64)

    # Pack initial params
    r0, t0 = T_to_rvec_t(T_b2u_init)
    x0 = np.hstack([r0, t0])

    # Residual function across ALL frames/points
    def residuals(x):
        rvec = x[:3]
        tvec = x[3:].reshape(3,1)
        # Build T_b2u once
        T_b2u = rvec_t_to_T(rvec, tvec)
        res_list = []
        for j, t2b in enumerate(T_t2b_list):
            T_t2u = t2b @ T_b2u      # target -> unknown (optical)
            R = T_t2u[:3,:3].astype(np.float64)
            # Orthonormalize rotation (keeps numerical drift in check)
            U, _, Vt = np.linalg.svd(R); R = U @ Vt
            if np.linalg.det(R) < 0: R[:, -1] *= -1
            rj, _ = cv2.Rodrigues(R)
            tj = T_t2u[:3,3].astype(np.float64).reshape(3,1)

            pred, _ = cv2.projectPoints(objp, rj, tj, K, D)
            diff = (pred.reshape(-1,2) - unknown_corners_list[j].reshape(-1,2)).astype(np.float64)
            res_list.append(diff.reshape(-1))  # stack (2*N,)
        return np.concatenate(res_list, axis=0)


    print(f"initial_residuals {residuals(x0)}")
    # Run robust LM (Huber)
    opt = least_squares(
        residuals, x0, method="lm" if huber_scale is None else "trf",
        loss="linear" if huber_scale is None else "huber",
        f_scale=huber_scale, verbose=verbose, max_nfev=200
    )

    r_opt = opt.x[:3]
    t_opt = opt.x[3:]
    T_opt = rvec_t_to_T(r_opt, t_opt)

    # Report pre/post RMSE
    def total_rmse(T_b2u):
        r = residuals(np.hstack(T_to_rvec_t(T_b2u))).reshape(-1,2)
        return float(np.sqrt(np.mean(np.sum(r**2, axis=1))))

    rmse_before = total_rmse(T_b2u_init)
    rmse_after  = total_rmse(T_opt)
    print(f"[BA] RMSE before: {rmse_before:.3f}px  after: {rmse_after:.3f}px  (delta: {rmse_before - rmse_after:.3f})")

    return T_opt, opt


if __name__ == "__main__":
    

    best_start = np.array(
        [
            [ 0.04795097, -0.99835908, -0.03130251,  0.03363864],
            [-0.72803621, -0.01347723, -0.68540619,  0.53896659],
            [ 0.68385962,  0.05565525, -0.72748781,  0.45695368],
            [ 0., 0., 0., 1.]
        ], dtype=np.float64
    )
    target_points = np.zeros((INNER_CORNERS[0]*INNER_CORNERS[1], 3), np.float32)
    target_points[:, :2] = np.mgrid[0:INNER_CORNERS[0], 0:INNER_CORNERS[1]].T.reshape(-1, 2)
    target_points *= float(SQUARE_SIZE_M)
    # Load intrinsics
    #K, D, distortion_model, width, height, R, P

    known_K, known_D, _, _, _, _, _ = load_intrinsics_npz(KNOWN_INTRINSICS_PATH)
    unknown_K, unknown_D, _, _, _, _, _ = load_intrinsics_npz(UNKNOWN_INTRINSICS_PATH)

    T_target2known_list = []
    T_target2unknown_list = []
    T_known2base_list = []
    unknown_corners_list = []

    npz_paths = sorted(glob.glob(os.path.join(npz_save_dir, "*.npz")))
    for npz_path in npz_paths:
        data = np.load(npz_path)
        kc = data["known_corners"]          # (N,1,2)
        uc = data["unknown_corners"]        # (N,1,2)

        T_k2b = data["T_known2base"]         # (4,4)
        T_t2k, known_reproj_error = target2cam_from_corners(kc, known_K, known_D, INNER_CORNERS, SQUARE_SIZE_M)
        T_t2u, unknown_reproj_error = target2cam_from_corners(uc, unknown_K, unknown_D, INNER_CORNERS, SQUARE_SIZE_M)

        if known_reproj_error > 1.0 or unknown_reproj_error > 1.0:
            print(f"[WARNING] Skipping frame {os.path.basename(npz_path)} due to high reproj. error "
                  f"(known: {known_reproj_error:.3f}px, unknown: {unknown_reproj_error:.3f}px)")
            continue
        unknown_corners_list.append(uc.astype(np.float64))
        T_target2known_list.append(T_t2k.astype(np.float64))
        T_target2unknown_list.append(T_t2u.astype(np.float64))
        T_known2base_list.append(T_k2b.astype(np.float64))



    best_T, opt = ba_refine_Tb2u(
        T_b2u_init=best_start,
        target_points=target_points,
        unknown_K=unknown_K,
        unknown_D=unknown_D,
        T_target2known_list=T_target2known_list,
        T_target2unknown_list=T_target2unknown_list,
        T_known2base_list=T_known2base_list,
        unknown_corners_list=unknown_corners_list,
        huber_scale=2.0,  # tweak 1.0–3.0 if needed
        verbose=2
    )


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