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
DATA_DIR = "/ros2_ws/src/vision_pipeline/vision_pipeline/figures/hand_in_eye_calibration"
assert os.path.exists(DATA_DIR), f"Data dir not found: {DATA_DIR}"
INTRINSICS_PATH = "/ros2_ws/src/vision_pipeline/vision_pipeline/figures/hand_in_eye_calibration/intrinsics.npz"
assert os.path.exists(INTRINSICS_PATH), f"Intrinsics file not found: {INTRINSICS_PATH}"

INNER_CORNERS = (10, 7)      # (cols, rows)
SQUARE_SIZE_M = 0.020         # 2cm

HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


# ---------------------- Utility functions ---------------------
def stack_T(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3)
    return T

# ----------------------- Visualization loaders -----------------------
def visualize_r_t(R_list, t_list, axis_len=0.1, draw_world=True, connect_trajectory=False, title=""):
    """
    Visualize a sequence of poses given by rotation matrices and translations.

    Parameters
    ----------
    R_list : list or (N,3,3) array-like
        Sequence of 3x3 rotation matrices. Each R maps local axes to world.
    t_list : list or (N,3) or (N,3,1) array-like
        Sequence of translations (world positions of the frame origin).
    axis_len : float, optional
        Length of the axis arrows for each frame.
    draw_world : bool, optional
        If True, draw world frame at the origin.
    connect_trajectory : bool, optional
        If True, connect the frame origins with a line.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes3D
    """
    R_arr = np.asarray(R_list, dtype=float)
    t_arr = np.asarray(t_list, dtype=float)

    # Normalize shapes: (N,3,3) and (N,3)
    if R_arr.ndim == 2 and R_arr.shape == (3, 3):
        R_arr = R_arr[None, ...]
    if t_arr.ndim == 1 and t_arr.shape == (3,):
        t_arr = t_arr[None, ...]
    if t_arr.ndim == 3 and t_arr.shape[-1] == 1:
        t_arr = t_arr[..., 0]  # (N,3,1) -> (N,3)

    # Basic validation
    if R_arr.ndim != 3 or R_arr.shape[1:] != (3, 3):
        raise ValueError(f"R_list must be (N,3,3); got {R_arr.shape}")
    if t_arr.ndim != 2 or t_arr.shape[1] != 3:
        raise ValueError(f"t_list must be (N,3) or (N,3,1); got {t_arr.shape}")
    if len(R_arr) != len(t_arr):
        raise ValueError(f"Lengths differ: len(R)={len(R_arr)} vs len(t)={len(t_arr)}")

    N = len(R_arr)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Optionally draw the world frame at origin
    if draw_world:
        origin = np.zeros(3)
        I = np.eye(3)
        _draw_frame(ax, origin, I, length=axis_len, label="W")

    # Draw each pose as a triad
    for i in range(N):
        t = t_arr[i]
        R = R_arr[i]
        _draw_frame(ax, t, R, length=axis_len, label=None)

    # Connect trajectory
    if connect_trajectory and N > 1:
        ax.plot(t_arr[:, 0], t_arr[:, 1], t_arr[:, 2], linewidth=1.5, alpha=0.8)

    # Axes cosmetics
    _set_axes_equal(ax, t_arr)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax

def _draw_frame(ax, t, R, length=0.1, label=None):
    """
    Draw a coordinate frame at position t with rotation R.
    X = R[:,0], Y = R[:,1], Z = R[:,2]
    """
    o = t.reshape(3)
    x_axis = R[:, 0] * length
    y_axis = R[:, 1] * length
    z_axis = R[:, 2] * length

    # X, Y, Z axes
    ax.plot([o[0], o[0] + x_axis[0]],
            [o[1], o[1] + x_axis[1]],
            [o[2], o[2] + x_axis[2]], linewidth=2, c='r')
    ax.plot([o[0], o[0] + y_axis[0]],
            [o[1], o[1] + y_axis[1]],
            [o[2], o[2] + y_axis[2]], linewidth=2, c='g')
    ax.plot([o[0], o[0] + z_axis[0]],
            [o[1], o[1] + z_axis[1]],
            [o[2], o[2] + z_axis[2]], linewidth=2, c='b')

    if label is not None:
        ax.text(o[0], o[1], o[2], label)

def _set_axes_equal(ax, points):
    """
    Make 3D plot axes have equal scale so that spheres look like spheres.
    Uses the extents of the provided points; falls back to a default cube if empty.
    """
    if points.size == 0:
        center = np.zeros(3)
        radius = 1.0
    else:
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        center = (mins + maxs) / 2.0
        radius = np.max(maxs - mins)
        if radius == 0:
            radius = 1.0
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(center[0] - radius/2, center[0] + radius/2)
    ax.set_ylim(center[1] - radius/2, center[1] + radius/2)
    ax.set_zlim(center[2] - radius/2, center[2] + radius/2)



# ----------------------- Data-specific loaders -----------------------
def load_intrinsics_npz(path: str):
    d = np.load(path, allow_pickle=True)
    K = d["K"].astype(float)
    D = d["D"].astype(float).ravel()
    distortion_model = str(d["distortion_model"]) if "distortion_model" in d else ""
    width = int(d["width"]) if "width" in d else None
    height = int(d["height"]) if "height" in d else None
    R = d["R"].astype(float) if "R" in d else None
    P = d["P"].astype(float) if "P" in d else None
    return K, D, distortion_model, width, height, R, P

def target2cam_from_corners(corners, K, D):
    """
    Estimate the 4x4 target->camera transform using solvePnP.

    Inputs:
        corners: np.ndarray shape (N,1,2) chessboard corners (OpenCV order).
        K, D           : camera matrix (3x3) and distortion coefficients.
        distortion_model : "pinhole" or "fisheye"
        pattern_size   : (cols, rows) so cols*rows == N
        

    Returns:
        T_target2cam : (4,4) homogeneous transform
    """
  
    # --- Build the target model points (board frame, Z=0 plane) ---
    cols, rows = INNER_CORNERS
    N = cols * rows
    assert corners.shape[0] == N, f"Expected {N} corners, got {corners.shape[0]}"
    objp = np.zeros((N, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    # print(f"[DEBUG] objp:\n{objp}")

    objp *= float(SQUARE_SIZE_M)


    # --- Prepare image points (N,2) float32 ---
    imgp = corners.reshape(-1, 2).astype(np.float32)
    # print(f"[DEBUG] imgp:\n{imgp}")

    # --- Solve PnP ---
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_IPPE)

    if not ok:
        raise RuntimeError("solvePnP failed to find a pose for the given corners.")
    
    corners_reproj, _ = cv2.projectPoints(
        objp, rvec, tvec, K, D
    )

    # Flatten to (N, 2)
    diff = (corners_reproj.reshape(-1, 2) - corners.reshape(-1, 2)).astype(np.float64)

    # Per-corner radial errors (L2 per point)
    per_corner = np.linalg.norm(diff, axis=1)

    # RMSE (radial, in pixels)
    rmse = float(np.sqrt(np.mean(per_corner**2)))


    R_target2cam, _ = cv2.Rodrigues(rvec)
    t_target2cam = tvec.reshape(3,1)

    # --- Pack into 4x4 ---
    T_target2cam = stack_T(R_target2cam, t_target2cam)

    return T_target2cam, rmse



def get_error(
    R_gripper2base: List[np.ndarray],
    t_gripper2base: List[np.ndarray],
    R_target2camera: List[np.ndarray],
    t_target2camera: List[np.ndarray],
    R_camera2gripper: np.ndarray,
    t_camera2gripper: np.ndarray,
    corners_list: List[np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
) -> float:
    """
    Compute overall reprojection RMSE [px] for a hand-eye solution.

    Inputs:
        R_gripper2base, t_gripper2base : lists of gripper->base rotations/translations (each 3x3, 3x1)
        R_target2camera, t_target2camera : lists of target->camera rotations/translations (each 3x3, 3x1)
        R_camera2gripper, t_camera2gripper : camera->gripper rotation/translation (3x3, 3x1) = X
        corners_list : list of detected chessboard corners, each (N,1,2)
        K, D         : camera intrinsics and distortion
    Returns:
        RMSE reprojection error over all frames & corners [pixels].
    """
    # --- Build transforms ---
    T_gripper2base = [stack_T(R, t) for R, t in zip(R_gripper2base, t_gripper2base)]
    T_target2camera = [stack_T(R, t) for R, t in zip(R_target2camera, t_target2camera)]
    T_camera2gripper= stack_T(R_camera2gripper, t_camera2gripper)
    T_gripper2camera = np.linalg.inv(T_camera2gripper)

    # --- Target->Base per frame ---
    T_target2base = [T_g2b @ T_camera2gripper @ T_t2c for T_g2b, T_t2c in zip(T_gripper2base, T_target2camera)]

    # ---Mean target->base pose as the reference ---
    R_list = [T[:3, :3] for T in T_target2base]
    t_stack = np.stack([T[:3, 3] for T in T_target2base], axis=0)
    visualize_r_t(R_list, t_stack, axis_len=0.005, title="Base->Target Poses")
    t_ref = t_stack.mean(axis=0)

    R_ref = SciRot.from_matrix(np.stack(R_list)).mean().as_matrix()

    T_ref_target2base = stack_T(R_ref, t_ref.reshape(3, 1))

    # --- Prepare chessboard model points (Z=0 plane in target frame) ---
    cols, rows = INNER_CORNERS
    N = cols * rows
    objp = np.zeros((N, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    objp *= float(SQUARE_SIZE_M)  # meters
    
    # --- Compute reference board points in base frame (meters) ---
    objp_h = np.hstack([objp, np.ones((N, 1), dtype=np.float32)])  # (N,4)
    P_ref_b = (T_ref_target2base @ objp_h.T).T[:, :3]  # (N,3) meters

    # --- For each frame: transform board points to base, compare to reference ---
    total_sq_err_m2 = 0.0
    total_pts = 0

    for T_t2b in T_target2base:
        P_b = (T_t2b @ objp_h.T).T[:, :3]  # (N,3) meters
        d = P_b - P_ref_b                  # (N,3) meters
        total_sq_err_m2 += float(np.sum(d ** 2))
        total_pts += N

    # RMSE in millimeters
    rmse_mm = 1000.0 * float(np.sqrt(total_sq_err_m2 / total_pts))
    return rmse_mm

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

    # visualize_r_t(R_gripper2base, t_gripper2base, axis_len=0.05, title="Gripper->Base Poses")
    # visualize_r_t(R_target2cam, t_target2cam, axis_len=0.05, title="Target->Camera Poses")
    # plt.show()


    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,  # lists of absolutes are expected here
        R_target2cam,  t_target2cam,
        method=HAND_EYE_METHODS["TSAI"]
    )
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

    error_mm = get_error(
        R_gripper2base, t_gripper2base,
        R_target2cam,  t_target2cam,
        R_cam2gripper, t_cam2gripper,
        corners_arr, K, D
    )
    print(f"TSAI")
    print(f"{error_mm=:.3f} mm RMS")
    print("\n\n")

    # Extract
    R = R_cam2gripper.copy()
    t = t_cam2gripper.copy()

    # Define the basis change for the CAMERA frame (old c -> new c')
    R_y_90 = SciRot.from_euler('y',  90, degrees=True).as_matrix()
    R_x_90 = SciRot.from_euler('x',  -90, degrees=True).as_matrix()
    C_c = R_x_90 @ R_y_90
    # Re-express cam->gripper in the new camera convention:
    R = R @ C_c.T     # right-multiply by C_c^T (source frame change)
    t = t             # translation unchanged for camera-frame change
    # Output values
    x, y, z = t.flatten().tolist()
    qx, qy, qz, qw = SciRot.from_matrix(R).as_quat()  # xyzw
    print(f"{x=:.6f}, {y=:.6f}, {z=:.6f}")
    print(f"{qx=}, {qy=}, {qz=}, {qw=}")

    print()
    print()
    print(f"'{x}', '{y}', '{z}',")
    print(f"'{qx}', '{qy}', '{qz}', '{qw}',")

    plt.show()

if __name__ == "__main__":
    main()
