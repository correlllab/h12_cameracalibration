import numpy as np
import os
import glob
import random
import cv2
import matplotlib.pyplot as plt
# ----------------------- SAVING utilities -----------------------
ready_to_save = False
def save_camera_info(camera_info, filepath):
    """
    Convert a ROS2 CameraInfo message into a NumPy .npz file.
    Stores K, D, R, P matrices and image size.
    """
    # Intrinsic matrix K (3x3)
    K = np.array(camera_info.k, dtype=np.float64).reshape(3, 3)

    # Distortion coefficients
    D = np.array(camera_info.d, dtype=np.float64)

    # Rectification matrix R (3x3)
    R = np.array(camera_info.r, dtype=np.float64).reshape(3, 3)

    # Projection matrix P (3x4)
    P = np.array(camera_info.p, dtype=np.float64).reshape(3, 4)

    # Save to .npz
    np.savez(
        filepath,
        width=camera_info.width,
        height=camera_info.height,
        distortion_model=camera_info.distortion_model,
        D=D,
        K=K,
        R=R,
        P=P,
        binning_x=camera_info.binning_x,
        binning_y=camera_info.binning_y,
        roi_x_offset=camera_info.roi.x_offset,
        roi_y_offset=camera_info.roi.y_offset,
        roi_height=camera_info.roi.height,
        roi_width=camera_info.roi.width,
        roi_do_rectify=camera_info.roi.do_rectify,
    )
def get_corners(rgb, target_dims):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, target_dims, flags)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners
def vis_and_save(camera_node, controller_node, ee_frame, camera_base_frame, camera_optical_frame, target_dims, save_dir):
    i = 0
    global ready_to_save
    last_t = np.eye(4)
    intrinsics_made = False
    extrinsics_made = False

    npz_save_dir = os.path.join(save_dir, 'npzs')
    os.makedirs(npz_save_dir, exist_ok=True)
    raw_save_dir = os.path.join(save_dir, 'raw')
    os.makedirs(raw_save_dir, exist_ok=True)
    annotated_save_dir = os.path.join(save_dir, 'annotated')
    os.makedirs(annotated_save_dir, exist_ok=True)

    intrinsic_path = os.path.join(npz_save_dir, 'intrinsics.npz')
    extrinsics_path = os.path.join(npz_save_dir, 'extrinsics.npz')


    while True:
        rgb, info, = camera_node.get_data()
        if not intrinsics_made and info is not None:
            save_camera_info(info, intrinsic_path)
            print(f"Saved intrinsics to {intrinsic_path}")
            intrinsics_made = True
        if not extrinsics_made:
            T = controller_node.get_tf(source_frame=camera_base_frame, target_frame=camera_optical_frame, timeout=1.0)
            if T is not None:
                np.savez(extrinsics_path, cam2optical=T)
                extrinsics_made = True
        transform = controller_node.get_tf(source_frame=ee_frame, target_frame="pelvis", timeout=1.0)
        if rgb is not None:
            h, w, _ = rgb.shape
            display_img = rgb.copy()
            d_T = float('inf')
        
            if transform is not None:
                d_T = np.linalg.norm(transform - last_t).mean()
                last_t = transform

            cv2.putText(display_img, f"{d_T:0.4f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            


            success, corners = get_corners(rgb)
            # print(f"{success=}, {(transform is None)=}")
            if success and transform is not None:
                cv2.drawChessboardCorners(display_img, target_dims, corners, success)
                stamp = f"{i=}"
                lin_diff = 0
                ang_diff = 0

                
                # print(f"{lin_diff=}, {ang_diff=}")
                if ready_to_save and d_T < 0.01:
                    cv2.imwrite(os.path.join(raw_save_dir, f"calib_{stamp}.png"), rgb)
                    cv2.imwrite(os.path.join(annotated_save_dir, f"calib_{stamp}.png"), display_img)
                    np.savez(os.path.join(npz_save_dir, f"calib_{stamp}.npz"), corners=corners, pose=transform)
                    print(f"Saved calib_{stamp}.png and calib_{stamp}.npz" )
                    i+=1
                    ready_to_save = False

            display_img = cv2.resize(display_img, (640, 480), interpolation = cv2.INTER_AREA)
            cv2.imshow("rgb", display_img)

        # quit on ESC
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            break
            
    cv2.destroyAllWindows()


def collect_control_loop(x,y,z,roll,target, controller_node, pose_func):
    saved = False
    global ready_to_save
    while not saved:
        T = pose_func(x, y, z, roll, target)
        # behavior_node.go_home(duration=5)
        print(f"\n\nMoving to x={x}, y={y}, z={z}, roll={roll}")
        print(f"Target: {target}")
        controller_node.send_arm_goal(right_mat=T, duration=5)
    
        cmd = input("Enter x y z r or dx dy dz dr or 'q' to quit, s to save, h for home, k to skip, tx, ty,tz to move the target point: ")
        if cmd.strip().lower() in ['q', 'quit', 'exit']:
            break
        if cmd == "s":
            ready_to_save = True
            n_tries = 0
            while ready_to_save and n_tries < 5: #wait for other thread to set it back to False
                time.sleep(0.1)
                n_tries+=1
            saved = True
            continue
            
        if cmd == "k":
            saved = True
            continue
        if cmd == "h":
            controller_node.go_home()
            continue
        
        value = input("Enter value: ")
        try:
            value = float(value)
        except ValueError:
            print("Invalid value. Please enter a numeric value.")
            continue
        if cmd.startswith('d'):
            if 'x' in cmd:
                x += value
            if 'y' in cmd:
                y += value
            if 'z' in cmd:
                z += value
            if 'r' in cmd:
                roll += value
        elif cmd.startswith('t'):
            if 'x' in cmd:
                target[0] += value
            if 'y' in cmd:
                target[1] += value
            if 'z' in cmd:
                target[2] += value
        else:
            if 'x' in cmd:
                x = value
            if 'y' in cmd:
                y = value
            if 'z' in cmd:
                z = value
            if 'r' in cmd:
                roll = value
    return x,y,z, roll, target


# ----------------------- SE3 utilities -----------------------
def stack_T(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3)
    return T
def inv_SE3(T):
    """Inverse of a rigid homogeneous transform (rotation+translation)."""
    R = T[:3,:3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Rt = R.T
    Ti[:3,:3] = Rt
    Ti[:3, 3] = -Rt @ t
    return Ti

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

def target2cam_from_corners(corners, K, D, target_dims, square_size_m):
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
    cols, rows = target_dims
    N = cols * rows
    assert corners.shape[0] == N, f"Expected {N} corners, got {corners.shape[0]}"
    objp = np.zeros((N, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    # print(f"[DEBUG] objp:\n{objp}")

    objp *= float(square_size_m)


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


def load_data(npz_dir, K, D, inner_corners, square_size_m):
    # Gather samples
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {npz_dir}")
    npz_files = [f for f in npz_files if os.path.basename(f) != "intrinsics.npz"]
    random.shuffle(npz_files)
    print(f"[INFO] Found {len(npz_files)} NPZ files in {npz_dir}")
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
        T_target2cam, error = target2cam_from_corners(corners, K, D, inner_corners, square_size_m)
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

    return (R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            corners_arr)