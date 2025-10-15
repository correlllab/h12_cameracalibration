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


def mean_rotation(Rs):
    M = sum(Rs)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:  # enforce proper rotation
        U[:,-1] *= -1
        Rm = U @ Vt
    return Rm

def rot_angle_deg(R):
    v = (np.trace(R) - 1.0) / 2.0
    v = np.clip(v, -1.0, 1.0)
    return np.degrees(np.arccos(v))

def eval_cam2base_constancy(
    R_gripper2base, t_gripper2base,
    R_cam2gripper,  t_cam2gripper,
    visualize = False
):
    # Build per-pose T_c2b from robot chain
    T_c2g = stack_T(R_cam2gripper, t_cam2gripper)

    T_c2b_list = []
    for R_g2b, t_g2b in zip(R_gripper2base, t_gripper2base):
        T_g2b = stack_T(R_g2b, t_g2b)
        T_c2b_i = T_g2b @ T_c2g   # cam->base for pose i
        T_c2b_list.append(T_c2b_i)
    if visualize:
        R_list = [T[:3, :3] for T in T_c2b_list]
        t_stack = np.stack([T[:3, 3] for T in T_c2b_list], axis=0)
        visualize_r_t(R_list, t_stack, axis_len=0.005, title="Base->Target Poses")

    # Reference: provided cam->base or the mean of estimates
    R_mean = mean_rotation([T[:3,:3] for T in T_c2b_list])
    t_mean = np.mean(np.vstack([T[:3,3] for T in T_c2b_list]), axis=0)
    T_ref  = stack_T(R_mean, t_mean)

    # Compute pose deltas wrt reference
    trans_err_mm = []
    rot_err_deg  = []
    T_ref_inv = np.linalg.inv(T_ref)
    for T in T_c2b_list:
        delta = T_ref_inv @ T
        rot_err_deg.append(rot_angle_deg(delta[:3,:3]))
        trans_err_mm.append( np.linalg.norm(delta[:3,3]) * 1000.0 )

    # RMS errors
    rms_mm  = float(np.sqrt(np.mean(np.square(trans_err_mm))))
    rms_deg = float(np.sqrt(np.mean(np.square(rot_err_deg))))
    return rms_mm, rms_deg

def grid_search_handeye(R_g2b, t_g2b, R_t2c, t_t2c, methods, print_all=True, try_rwhe=True):
    """
    Try small permutations of input order/inverses & method for calibrateHandEye,
    print a full results table, and optionally run Robot-World–Hand-Eye as well.

    Returns:
        results: list[dict] sorted by rms_mm asc for calibrateHandEye
        rwhe_results: list[dict] (may be empty) for calibrateRobotWorldHandEye
    """
    def invert_list(Rs, ts):
        Rs_i, ts_i = [], []
        for R, t in zip(Rs, ts):
            R_i = R.T
            t_i = -R_i @ t
            Rs_i.append(R_i); ts_i.append(t_i)
        return Rs_i, ts_i

    # ----- Build the combos to test (calibrateHandEye) -----
    combos = []
    combos.append(("AS_IS", R_g2b, t_g2b, R_t2c, t_t2c))                         # A
    R_b2g, t_b2g = invert_list(R_g2b, t_g2b)
    combos.append(("INV_ROBOT", R_b2g, t_b2g, R_t2c, t_t2c))                     # B
    R_c2t, t_c2t = invert_list(R_t2c, t_t2c)
    combos.append(("INV_VISION", R_g2b, t_g2b, R_c2t, t_c2t))                     # C
    combos.append(("INV_BOTH", R_b2g, t_b2g, R_c2t, t_c2t))                       # D
    combos.append(("SWAP", R_t2c, t_t2c, R_g2b, t_g2b))                           # E
    combos.append(("SWAP_INV_BOTH", R_c2t, t_c2t, R_b2g, t_b2g))                  # F

    results = []
    for combo_name, Ra, ta, Rb, tb in combos:
        for mname, mflag in methods.items():
            try:
                R_c2g, t_c2g = cv2.calibrateHandEye(Ra, ta, Rb, tb, method=mflag)
                rms_mm, rms_deg = eval_cam2base_constancy(R_g2b, t_g2b, R_c2g, t_c2g)
            except cv2.error:
                R_c2g, t_c2g = None, None
                rms_mm, rms_deg = float("inf"), float("inf")
            results.append(dict(
                combo_name=combo_name,
                method=mname,
                rms_mm=rms_mm,
                rms_deg=rms_deg,
                R_c2g=R_c2g,
                t_c2g=t_c2g
            ))

    # Sort by best (lowest translational constancy error)
    results.sort(key=lambda d: d["rms_mm"])

    # ----- Optional: Robot-World–Hand-Eye sweep (eye-to-hand) -----
    rwhe_results = []
    if try_rwhe:
        # Build available RWHE method flags safely
        rwhe_method_names = ["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"]
        rwhe_methods = {
            "SHAH": cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
            "LI": cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI
        }
        rwhe_methods = {k: v for k, v in rwhe_methods.items() if v is not None}

        for mname, mflag in rwhe_methods.items():
            try:
                # Inputs are the absolutes you already collected
                R_w2c, t_w2c, R_t2g, t_t2g = cv2.calibrateRobotWorldHandEye(
                    R_g2b, t_g2b,   # hand (gripper->base) absolutes
                    R_t2c, t_t2c,   # target->camera absolutes
                    method=mflag
                )
                # Evaluate: camera->base constancy (invert world->camera)
                R_c2w = R_w2c.T
                t_c2w = -(R_c2w @ t_w2c)
                rms_mm, rms_deg = eval_cam2base_constancy(R_g2b, t_g2b, R_c2w, t_c2w)
                rwhe_results.append(dict(
                    method=f"RWHE_{mname}",
                    rms_mm=rms_mm, rms_deg=rms_deg,
                    R_w2c=R_w2c, t_w2c=t_w2c,
                    R_t2g=R_t2g, t_t2g=t_t2g
                ))
            except cv2.error:
                rwhe_results.append(dict(
                    method=f"RWHE_{mname}",
                    rms_mm=float("inf"), rms_deg=float("inf"),
                    R_w2c=None, t_w2c=None, R_t2g=None, t_t2g=None
                ))
        rwhe_results.sort(key=lambda d: d["rms_mm"])
        print(f"{rwhe_results=}")

    # ----- Printing -----
    if print_all:
        print("\n=== calibrateHandEye candidates (sorted by RMS mm) ===")
        for r in results:
            print(f"{r['combo_name']:>16} | {r['method']:<10} | "
                  f"{r['rms_mm']:8.1f} mm | {r['rms_deg']:7.2f} deg")

        print("\n--- SWAP-only (useful when camera is fixed & board on wrist) ---")
        for r in results:
            if r["combo_name"] in ("SWAP", "SWAP_INV_BOTH"):
                print(f"{r['combo_name']:>16} | {r['method']:<10} | "
                      f"{r['rms_mm']:8.1f} mm | {r['rms_deg']:7.2f} deg")

        if rwhe_results:
            print("\n=== Robot-World–Hand-Eye (fixed camera) ===")
            for r in rwhe_results:
                print(f"{r['method']:>16} | {r['rms_mm']:8.1f} mm | {r['rms_deg']:7.2f} deg")

        best = results[0]
        print("\nPicked (calibrateHandEye):",
              best["combo_name"], best["method"],
              f"-> {best['rms_mm']:.1f} mm / {best['rms_deg']:.2f} deg")
        if rwhe_results:
            best_rwhe = rwhe_results[0]
            print("Picked (RWHE):",
                  best_rwhe["method"],
                  f"-> {best_rwhe['rms_mm']:.1f} mm / {best_rwhe['rms_deg']:.2f} deg")

    return results, rwhe_results


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
        # T_target2cam = np.linalg.inv(T_target2cam)
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
        # T_gripper2base = np.linalg.inv(T_gripper2base)
        R_gripper2base.append(T_gripper2base[:3, :3])
        t_gripper2base.append(T_gripper2base[:3, 3].reshape(3,1))

        
    print(f"[INFO] Rejected {rejected} samples due to high reprojection error.")
    assert len(R_gripper2base) == len(t_gripper2base) == len(R_target2cam) == len(t_target2cam)
    print(f"[INFO] Loaded {len(R_gripper2base)} samples")
    print(f"[INFO] Example shape: {R_gripper2base[0].shape=}, {t_gripper2base[0].shape=} {R_target2cam[0].shape=}, {t_target2cam[0].shape=}")

    best_T = np.eye(4)
    best_error = float('inf')
    ransac_iters = 100
    sample_n = 7


    results = grid_search_handeye(R_gripper2base, t_gripper2base,
                              R_target2cam,  t_target2cam,
                              HAND_EYE_METHODS)




    # for i in range(ransac_iters):

    #     sample_idxs = random.sample(range(len(R_gripper2base)), sample_n)
    #     R_gripper2base_sample = [R_gripper2base[i] for i in sample_idxs]
    #     t_gripper2base_sample = [t_gripper2base[i] for i in sample_idxs]
    #     R_target2cam_sample = [R_target2cam[i] for i in sample_idxs]
    #     t_target2cam_sample = [t_target2cam[i] for i in sample_idxs]

    #     R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    #         R_gripper2base_sample, t_gripper2base_sample,  # lists of absolutes are expected here
    #         R_target2cam_sample,  t_target2cam_sample,
    #         method=HAND_EYE_METHODS["TSAI"]
    #     )
    #     T_cam2gripper = np.eye(4)
    #     T_cam2gripper[:3, :3] = R_cam2gripper
    #     T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

    #     error_mm, _ = eval_cam2base_constancy(
    #         R_gripper2base, t_gripper2base,
    #         R_cam2gripper, t_cam2gripper,
    #         visualize=False
    #     )
    #     print(f"{i}: {error_mm=:.3f} mm RMS")
    #     print("\n\n")
    #     if error_mm < best_error:
    #         best_error = error_mm
    #         best_T = T_cam2gripper.copy()

    # print(f"[RESULT] Best reprojection error: {best_error:.3f} mm RMS")
    
    # eval_cam2base_constancy(
    #     R_gripper2base, t_gripper2base,
    #     best_T[:3, :3], best_T[:3, 3].reshape(3,1),
    #     visualize=True
    # )




    # T_camOpt2gripper = best_T.copy()

    # T_cam2gripper = T_camOpt2gripper @ rs2optical
    

    # R_final = T_cam2gripper[:3, :3]
    # t_final = T_cam2gripper[:3,  3]


    # # Output values
    # x, y, z = t_final.flatten().tolist()
    # qx, qy, qz, qw = SciRot.from_matrix(R_final).as_quat()  # xyzw
    # print(f"{x=:.6f}, {y=:.6f}, {z=:.6f}")
    # print(f"{qx=}, {qy=}, {qz=}, {qw=}")

    # print()
    # print()
    # print(f"'{x}', '{y}', '{z}',")
    # print(f"'{qx}', '{qy}', '{qz}', '{qw}',")

    # plt.show()

if __name__ == "__main__":
    main()
