import time
import shutil
import numpy as np
import os
import sys
import cv2
file_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(file_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
save_dir = os.path.join(data_dir, 'handineye_calibration')


input(f"press anything to delete {save_dir} and continue")
shutil.rmtree(save_dir, ignore_errors=True)
os.makedirs(save_dir, exist_ok=True)


npz_save_dir = os.path.join(save_dir, 'npzs')
os.makedirs(npz_save_dir, exist_ok=True)
raw_save_dir = os.path.join(save_dir, 'raw')
os.makedirs(raw_save_dir, exist_ok=True)
annotated_save_dir = os.path.join(save_dir, 'annotated')
os.makedirs(annotated_save_dir, exist_ok=True)


if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

from controller import ControllerNode
from camerainterface import CameraSubscriber
import threading



TARGET_DIMS = (10, 7)


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
def get_corners(rgb):
    global reverse_corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, TARGET_DIMS, flags)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    if reverse_corners and ret:
        corners = corners[::-1]
    return ret, corners


ready_to_save = False
reverse_corners = False
def vis_and_save(camera_node, controller_node, intrinsic_path, extrinsics_path):
    i = 0
    global ready_to_save
    last_t = np.eye(4)
    intrinsics_made = False
    extrinsics_made = False
    while True:
        rgb, info, = camera_node.get_data()
        if not intrinsics_made and info is not None:
            save_camera_info(info, intrinsic_path)
            print(f"Saved intrinsics to {intrinsic_path}")
            intrinsics_made = True
        if not extrinsics_made:
            transform = controller_node.get_tf(source_frame="left_hand_link", target_frame="left_hand_color_optical_frame", timeout=1.0)
            if transform is not None:
                np.savez(extrinsics_path, cam2optical=transform)
                extrinsics_made = True
        transform = controller_node.get_tf(source_frame="left_wrist_yaw_link", target_frame="pelvis", timeout=1.0)
        if rgb is not None:
            h, w, _ = rgb.shape
            display_img = rgb.copy()
            d_T = np.linalg.norm(transform - last_t).mean()
            
            cv2.putText(display_img, f"{d_T:0.4f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            
            if transform is not None:
                last_t = transform

            success, corners = get_corners(rgb)
            if success and transform is not None:
                cv2.drawChessboardCorners(display_img, TARGET_DIMS, corners, success)
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

def get_lookat_matrix(x,y,z,roll, target):
    pos = np.array([x, y, z], dtype=float)
    dir_vec = target - pos
    norm = np.linalg.norm(dir_vec)
    x_axis = dir_vec / norm

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, x_axis)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
        
    y0 = up - np.dot(up, x_axis) * x_axis
    y0 /= np.linalg.norm(y0)
    z0 = np.cross(x_axis, y0)
    z0 /= np.linalg.norm(z0)

    roll = np.deg2rad(roll)
    c, s = np.cos(roll), np.sin(roll)

    # rotate y0,z0 around x_axis by 'roll' (Rodrigues)
    # v_rot = v*c + (k×v)*s + k*(k·v)*(1-c), here k = x_axis
    k = x_axis
    y_axis = y0 * c + np.cross(k, y0) * s + k * np.dot(k, y0) * (1 - c)  # dot=0, so last term is 0
    z_axis = z0 * c + np.cross(k, z0) * s + k * np.dot(k, z0) * (1 - c)  # dot=0, so last term is 0


    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    return T

def collect(x,y,z,roll,target, controller_node):
    saved = False
    global ready_to_save
    global reverse_corners
    while not saved:
        T = get_lookat_matrix(x, y, z, roll, target)
        # behavior_node.go_home(duration=5)
        print(f"\n\nMoving to x={x}, y={y}, z={z}, roll={roll}")
        print(f"Target: {target}")
        controller_node.send_arm_goal(left_mat=T, duration=5)
    
        cmd = input("Enter x y z r or dx dy dz dr or 'q' to quit, r to reverse the corner ordering, s to save, h for home, k to skip, tx, ty,tz to move the target point: ")
        if cmd.strip().lower() in ['q', 'quit', 'exit']:
            break
        if cmd == "r":
            reverse_corners = not reverse_corners
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

def main():
    controller_node = ControllerNode()
    camera_node = CameraSubscriber("/realsense/left_hand")
    intrinsic_path = os.path.join(save_dir, "intrinsics.npz")
    extrinsics_path = os.path.join(save_dir, "extrinsics.npz")
    vis_thread = threading.Thread(target=vis_and_save, args=(camera_node, controller_node, intrinsic_path, extrinsics_path))
    vis_thread.start()
    time.sleep(1)
    print()
    print("camera initialized")

    target_location = [0.1, 0.85, 0.2]
    target = np.array(target_location, dtype=float)
    configs = [
        #straight y
        [0, 0.1, 0, 0],
        [0, 0.3, 0, 0],
        [0, 0.6, 0, 0],

        #y at x extreme
        [0.3, 0.1, 0, 0],
        [0.3, 0.3, 0, 0],
        [0.3, 0.6, 0, 0],
        
        #y at -x extreme
        [-0.3, 0.1, 0, 0],
        [-0.3, 0.3, 0, 0],
        [-0.3, 0.6, 0, 0],
        
        #y at z extreme
        [0.0, 0.1, 0.3, 0],
        [0.0, 0.3, 0.3, 0],
        [0.0, 0.6, 0.3, 0],
       
        #y at -z extreme
        [0.0, 0.1, -0.3, 0],
        [0.0, 0.3, -0.3, 0],
        [0.0, 0.6, -0.3, 0],

        #y at x and z extreme
        [0.3, 0.1, 0.3, 0],
        [0.3, 0.3, 0.3, 0],
        [0.3, 0.6, 0.3, 0],

        #y at -x and -z extreme
        [-0.3, 0.1, -0.3, 0],
        [-0.3, 0.3, -0.3, 0],
        [-0.3, 0.6, -0.3, 0],

        #y at x and -z extreme
        [0.3, 0.1, -0.3, 0],
        [0.3, 0.3, -0.3, 0],
        [0.3, 0.6, -0.3, 0],

        #y at -x and z extreme
        [-0.3, 0.1, 0.3, 0],
        [-0.3, 0.3, 0.3, 0],
        [-0.3, 0.6, 0.3, 0],


        #y at 45 roll
        [0, 0.1, 0, 45],
        [0, 0.3, 0, 45],
        [0, 0.6, 0, 45],

        #y at -45 roll
        [0, 0.1, 0, -45],
        [0, 0.3, 0, -45],
        [0, 0.6, 0, -45],

        #y at roll and -x extremes
        [-0.3, 0.1, 0, 45],
        [-0.3, 0.3, 0, 67],
        [-0.3, 0.5, 0, 90],

        #y at roll and x extremes
        [0.3, 0.1, 0, -45],
        [0.3, 0.3, 0, -67],
        [0.3, 0.5, 0, -90],

        # below is from gpt
        # --- Elevated (top-down-ish), mid-range
        [ 0.35, 0.30,  0.35,   0],
        [-0.35, 0.30,  0.35,   0],
        [ 0.35, 0.50,  0.35,  45],
        [-0.35, 0.50,  0.35,  45],

        # --- Low (bottom-up-ish), mid-range
        [ 0.35, 0.30, -0.35,   0],
        [-0.35, 0.30, -0.35,   0],
        [ 0.35, 0.50, -0.35,  45],
        [-0.35, 0.50, -0.35, -45],

        # --- Oblique diagonals (x and z both nonzero), mid-range
        [ 0.25, 0.40,  0.25,  30],
        [-0.25, 0.40,  0.25, -30],
        [ 0.25, 0.40, -0.25, -30],
        [-0.25, 0.40, -0.25,  30],

        # --- Farther range to excite intrinsics/distortion
        [ 0.20, 0.80,  0.20,   0],
        [-0.20, 0.80,  0.20,   0],
        [ 0.20, 0.80, -0.20,  60],
        [-0.20, 0.80, -0.20, -60],

        # --- Closer range (near), gentle elevation
        [ 0.15, 0.05,  0.15,   0],
        [-0.15, 0.05,  0.15,   0],
        [ 0.15, 0.05, -0.15,  90],
        [-0.15, 0.05, -0.15, -90],

        # --- Symmetric side sweeps, small elevation, multiple rolls
        [ 0.40, 0.20,  0.10,   0],
        [ 0.40, 0.20,  0.10,  45],
        [-0.40, 0.20,  0.10,   0],
        [-0.40, 0.20,  0.10, -45],

        # --- Higher elevation, stronger roll variety
        [ 0.00, 0.45,  0.45,   0],
        [ 0.00, 0.45,  0.45,  67],
        [ 0.00, 0.45,  0.45, -67],

        # --- Lower elevation, stronger roll variety
        [ 0.00, 0.45, -0.45,   0],
        [ 0.00, 0.45, -0.45,  67],
        [ 0.00, 0.45, -0.45, -67],
    ]
    for i, (x, y, z, roll) in enumerate(configs):
        print(f"\n\n{i+1}/{len(configs)} New position: x={x}, y={y}, z={z}, roll={roll}")
        collect(x,y,z,roll,target, controller_node)



    print(f"\n\nAll done! Returning home")
    controller_node.go_home(duration=10)
    exit(0)
        

if __name__ == "__main__":
    main()