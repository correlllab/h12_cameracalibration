import numpy as np
import os
import glob
import random
# import cv2
# import matplotlib.pyplot as plt
import time
import threading
# ----------------------- SAVING utilities -----------------------
save_request = threading.Event()
save_done = threading.Event()
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
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCornersSB(gray, target_dims, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners
def vis_and_save(controller_node, camera_nodes, ee_frame, base_frame, camera_base_frames, camera_optical_frames, target_dims, save_dir):
    print("in vis and save")
    save_count = 0
    global save_request
    global save_done
    last_H = np.eye(4)
    
    assert len(camera_nodes) == len(camera_base_frames) == len(camera_optical_frames)
    intrinsics_made_list = [False] * len(camera_nodes)
    extrinsics_made_list = [False] * len(camera_nodes)

    npz_save_dir = os.path.join(save_dir, 'npzs')
    os.makedirs(npz_save_dir, exist_ok=True)
    raw_save_dir = os.path.join(save_dir, 'raw')
    os.makedirs(raw_save_dir, exist_ok=True)
    annotated_save_dir = os.path.join(save_dir, 'annotated')
    os.makedirs(annotated_save_dir, exist_ok=True)

    intrinsics_paths = [os.path.join(save_dir, f'intrinsics_{i}.npz') for i in range(len(camera_nodes))]
    extrinsics_paths = [os.path.join(save_dir, f'extrinsics_{i}.npz') for i in range(len(camera_nodes))]

    # print("begining loop")
    while True:
        # print("looped")
        camera_data_list = [camera_node.get_data() for camera_node in camera_nodes]
        rgb_list = [data[0] for data in camera_data_list]
        info_list = [data[1] for data in camera_data_list]
        for i in range(len(camera_nodes)):
            intrinsics_made = intrinsics_made_list[i]
            extrinsics_made = extrinsics_made_list[i]
            info = info_list[i]
            if not intrinsics_made and info is not None:
                save_camera_info(info, intrinsics_paths[i])
                print(f"Saved intrinsics to {intrinsics_paths[i]}")
                intrinsics_made_list[i] = True
            if not extrinsics_made:
                H = controller_node.get_tf(source_frame=camera_base_frames[i], target_frame=camera_optical_frames[i], timeout=1.0)
                if H is not None:
                    np.savez(extrinsics_paths[i], H_cameraoptical_camerabase=H)
                    extrinsics_made_list[i] = True
        
        H_base_ee = controller_node.get_tf(source_frame=ee_frame, target_frame=base_frame, timeout=1.0)
        d_H = float('inf')
        if H_base_ee is not None:
            d_H = np.linalg.norm(H_base_ee - last_H)
            last_H = H_base_ee

        all_rgbs_good = True
        for rgb in rgb_list:
            if rgb is None:
                all_rgbs_good = False

        if all_rgbs_good:
            display_imgs = [rgb.copy() for rgb in rgb_list]
            for display_img in display_imgs:
                cv2.putText(display_img, f"{d_H:0.4f}",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)


            corner_results = [get_corners(rgb, target_dims) for rgb in rgb_list]
            all_corners_found = True
            for i, (success, corners) in enumerate(corner_results):
                if not success:
                    all_corners_found = False
                else:
                    cv2.drawChessboardCorners(display_imgs[i], target_dims, corners, success)

            save_requested = save_request.is_set()
            if all_corners_found and save_requested and d_H < 0.0001:
                for i in range(len(camera_nodes)):
                    cv2.imwrite(os.path.join(raw_save_dir, f"calib_{save_count}_cam_{i}.png"), rgb_list[i])
                    cv2.imwrite(os.path.join(annotated_save_dir, f"calib_{save_count}_cam_{i}.png"), display_imgs[i])
                    np.savez(os.path.join(npz_save_dir, f"calib_{save_count}_cam_{i}.npz"), corners=corner_results[i][1], H_base_ee=H_base_ee)

                save_count += 1
                save_request.clear()
                save_done.set()
            for i in range(len(display_imgs)):
                display_img = cv2.resize(display_imgs[i], (640, 480), interpolation = cv2.INTER_AREA)
                cv2.imshow(f"rgb_camera_{i}", display_img)
                # print("imshow called")
        # quit on ESC
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            break
            
    cv2.destroyAllWindows()
def collect_control_loop(x,y,z,roll,target, controller_node, pose_func, use_right = False):
    saved = False
    global save_request
    global save_done
    while not saved:
        H = pose_func(x, y, z, roll, target)
        # behavior_node.go_home(duration=5)
        print(f"\n\nMoving to x={x}, y={y}, z={z}, roll={roll}")
        print(f"Target: {target}")
        if use_right:
            controller_node.send_arm_goal(right_mat=H, duration=5)
        else:
            controller_node.send_arm_goal(left_mat=H, duration=5)

        cmd = input("Enter x y z r or dx dy dz dr or 'q' to quit, s to save, h for home, k to skip, tx, ty,tz to move the target point: ")
        if cmd.strip().lower() in ['q', 'quit', 'exit']:
            break
        if cmd == "s":
            save_done.clear()
            save_request.set()
            ok = save_done.wait(timeout=3.0)
            save_request.clear()
            if ok:
                print("[OK] Saved sample.")
                saved = True
            else:
                print("[WARN] Save failed (no corners or robot moved). Try again.")
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
def predict_corners(H_c_t, inner_corners, square_size_m, K, D):
    objp = get_target_points(inner_corners, square_size_m)
    rvec, tvec = cv2.Rodrigues(H_c_t[:3, :3])[0], H_c_t[:3, 3].reshape(3, 1)

    imgpts, _ = cv2.projectPoints(
        objp,
        rvec,
        tvec,
        K,
        D
    )  # imgpts: (N,1,2)
    return imgpts
# ----------------------- SE3 utilities -----------------------
def stack_H(R, t):
    H = np.eye(4)
    H[:3,:3] = R
    H[:3, 3] = t.reshape(3)
    return H
def inv_SE3(H):
    """Inverse of a rigid homogeneous transform (rotation+translation)."""
    R = H[:3,:3]
    t = H[:3, 3]
    Hi = np.eye(4)
    Rt = R.T
    Hi[:3,:3] = Rt
    Hi[:3, 3] = -Rt @ t
    return Hi
def get_target_points(target_dims, square_size_m):
    # --- Build the target model points (board frame, Z=0 plane) ---
    cols, rows = target_dims
    N = cols * rows
    objp = np.zeros((N, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    # print(f"[DEBUG] objp:\n{objp}")

    objp *= float(square_size_m)
    return objp
# ----------------------- Visualization loaders -----------------------
def visualize_r_t(R_list, t_list, axis_len=0.1, draw_world=True, connect_trajectory=False, title="", show=True):
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
    if show:
        plt.show()
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

def visualize_corners(imgpath_list, true_corner_list, pred_corner_list, title="", show=True):
    assert len(imgpath_list) == len(true_corner_list) == len(pred_corner_list)
    sidelength = np.ceil(np.sqrt(len(imgpath_list)))
    fig, axes = plt.subplots(int(sidelength), int(sidelength), figsize=(30, 30))
    axes = axes.flatten()
    for img_path, true_corners, pred_corners, ax in zip(imgpath_list, true_corner_list, pred_corner_list, axes):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.scatter(true_corners[:,0,0], true_corners[:,0,1], c='g', marker='o', label='True Corners')
        ax.scatter(pred_corners[:,0,0], pred_corners[:,0,1], c='r', marker='x', label='Predicted Corners')
        ax.axis('off')
        ax.legend()
    fig.tight_layout()
    fig.suptitle(title, fontsize=20)
    if show:
        plt.show()

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

def H_cam_target_from_corners(corners, K, D, target_dims, square_size_m):
    """
    Estimate the 4x4 target->camera transform using solvePnP.

    Inputs:
        corners: np.ndarray shape (N,1,2) chessboard corners (OpenCV order).
        K, D           : camera matrix (3x3) and distortion coefficients.
        distortion_model : "pinhole" or "fisheye"
        pattern_size   : (cols, rows) so cols*rows == N
        

    Returns:
        H_cam_target : (4,4) homogeneous transform
    """
  
    
    objp = get_target_points(target_dims, square_size_m)
    # --- Prepare image points (N,2) float32 ---
    imgp = corners.reshape(-1, 2).astype(np.float32)
    # print(f"[DEBUG] imgp:\n{imgp}")

    # --- Solve PnP ---
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_IPPE)
    R = cv2.Rodrigues(rvec)[0]
    H_c_t = stack_H(R, tvec)

    if not ok:
        raise RuntimeError("solvePnP failed to find a pose for the given corners.")
    
    predicted_corners = predict_corners(H_c_t, target_dims, square_size_m, K, D)

    # Flatten to (N, 2)
    diff = (predicted_corners.reshape(-1, 2) - corners.reshape(-1, 2)).astype(np.float64)

    # Per-corner radial errors (L2 per point)
    per_corner = np.linalg.norm(diff, axis=1)

    # RMSE (radial, in pixels)
    rmse = float(np.sqrt(np.mean(per_corner**2)))


    R_cam_target, _ = cv2.Rodrigues(rvec)
    t_cam_target = tvec.reshape(3,1)

    # --- Pack into 4x4 ---
    H_cam_target = stack_H(R_cam_target, t_cam_target)
        


    return H_cam_target, rmse

def load_data(npz_dir, K, D, inner_corners, square_size_m, img_dir, display = True):
    # Gather samples
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {npz_dir}")
    random.shuffle(npz_files)
    print(f"[INFO] Found {len(npz_files)} NPZ files in {npz_dir}")
    R_base_gripper = []
    t_base_gripper = []
    R_cam_target = []
    t_cam_target = []
    corners_arr = []
    rejected = 0
    img_path_arr = []

    for f in npz_files:
        data = np.load(f)
        corners = data["corners"]
        print(" -", os.path.basename(f))
        H_cam_target, error = H_cam_target_from_corners(corners, K, D, inner_corners, square_size_m)
        # H_cam_target = np.linalg.inv(H_cam_target)
        print(f"  Reprojection error rmse: {error:.3f} px")
        if error > 1.5:
            print(f"  [WARNING] High reprojection error {error:.3f} px, rejecting this sample.")
            rejected += 1
            continue
        print()
        corners_arr.append(corners)

        R_cam_target.append(H_cam_target[:3, :3])
        t_cam_target.append(H_cam_target[:3, 3].reshape(3,1))
        H_base_ee = data["H_base_ee"]
        # H_base_ee = np.linalg.inv(H_base_ee)
        R_base_gripper.append(H_base_ee[:3, :3])
        t_base_gripper.append(H_base_ee[:3, 3].reshape(3,1))

        img_path = os.path.join(img_dir, os.path.basename(f).replace('.npz', '.png'))
        assert os.path.exists(img_path), f"Image file not found: {img_path}"
        img_path_arr.append(img_path)
        
    print(f"[INFO] Rejected {rejected} samples due to high reprojection error.")
    assert len(R_base_gripper) == len(t_base_gripper) == len(R_cam_target) == len(t_cam_target)
    print(f"[INFO] Loaded {len(R_base_gripper)} samples")
    print(f"[INFO] Example shape: {R_base_gripper[0].shape=}, {t_base_gripper[0].shape=} {R_cam_target[0].shape=}, {t_cam_target[0].shape=}")

    if display:
        visualize_r_t(R_base_gripper, t_base_gripper, title="Loaded Base to Gripper Poses", show = False)
        visualize_r_t(R_cam_target, t_cam_target, title="Loaded Camera to Target Poses", show = False)
        visualize_corners(
            img_path_arr,
            corners_arr,
            [predict_corners(stack_H(R, t), inner_corners, square_size_m, K, D) for R, t in zip(R_cam_target, t_cam_target)],
            title="PNP reprojection"
        )

    return (R_base_gripper, t_base_gripper,
            R_cam_target, t_cam_target,
            corners_arr, img_path_arr)