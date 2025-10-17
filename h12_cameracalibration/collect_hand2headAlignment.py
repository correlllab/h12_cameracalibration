import glob
import itertools
import time
import shutil
import numpy as np
import os
import sys
import cv2
from scipy.spatial.transform import Rotation as SciRot

file_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(file_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
save_dir = os.path.join(data_dir, 'hand2eye_alignment')


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

KNOWN_OPTICAL_FRAME = "left_hand_color_optical_frame"
UNKNOWN_OPTICAL_FRAME = "head_color_optical_frame"
UNKNOWN_BASE_FRAME = "head_link"
BASEFRAME = "pelvis"

TARGET_DIMS = (10, 7)
SQUARE_SIZE_M = 0.02


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
saved_data = []
def vis_and_save(known_camera_node, unknown_camera_node, controller_node, known_intrinsic_path, unknown_intrinsic_path, unknown_extrinsics_path):
    i = 0
    global ready_to_save
    global saved_data
    last_known_t = np.eye(4)
    known_intrinsics_made = False
    unknown_intrinsics_made = False
    unknown_extrinsics_made = False
    while True:
        known_rgb, known_info, = known_camera_node.get_data()
        unknown_rgb, unknown_info, = unknown_camera_node.get_data()
        #save intrinsics
        if not known_intrinsics_made and known_info is not None:
            save_camera_info(known_info, known_intrinsic_path)
            print(f"Saved intrinsics to {known_intrinsic_path}")
            known_intrinsics_made = True
        if not unknown_intrinsics_made and unknown_info is not None:
            save_camera_info(unknown_info, unknown_intrinsic_path)
            print(f"Saved intrinsics to {unknown_intrinsic_path}")
            unknown_intrinsics_made = True
        
        #save extrinsics
        if not unknown_extrinsics_made:
            T = controller_node.get_tf(source_frame=UNKNOWN_BASE_FRAME, target_frame=UNKNOWN_OPTICAL_FRAME, timeout=1.0)
            if T is not None:
                np.savez(unknown_extrinsics_path, cam2optical=T)
                unknown_extrinsics_made = True

        T_known2base = controller_node.get_tf(source_frame=KNOWN_OPTICAL_FRAME, target_frame=BASEFRAME, timeout=1.0)
        if known_rgb is not None and unknown_rgb is not None:
            known_display_img = known_rgb.copy()
            unknown_display_img = unknown_rgb.copy()
            d_T = np.inf
            if T_known2base is not None:
                d_T = np.linalg.norm(T_known2base - last_known_t).mean()
                last_known_t = T_known2base

            cv2.putText(known_display_img, f"{d_T:0.4f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            cv2.putText(unknown_display_img, f"{d_T:0.4f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            


            known_success, known_corners = get_corners(known_rgb)
            unknown_success, unknown_corners = get_corners(unknown_rgb)
            # print(f"{success=}, {(transform is None)=}")
            if known_success and unknown_success:
                cv2.drawChessboardCorners(known_display_img, TARGET_DIMS, known_corners, known_success)
                cv2.drawChessboardCorners(unknown_display_img, TARGET_DIMS, unknown_corners, unknown_success)
                stamp = f"{i=}"

                # print(f"{lin_diff=}, {ang_diff=}")
                if ready_to_save and d_T < 0.01 and T_known2base is not None:
                    cv2.imwrite(os.path.join(raw_save_dir, f"calib_{stamp}_known.png"), known_rgb)
                    cv2.imwrite(os.path.join(raw_save_dir, f"calib_{stamp}_unknown.png"), unknown_rgb)
                    cv2.imwrite(os.path.join(annotated_save_dir, f"calib_{stamp}_known.png"), known_display_img)
                    cv2.imwrite(os.path.join(annotated_save_dir, f"calib_{stamp}_unknown.png"), unknown_display_img)
                    np.savez(os.path.join(npz_save_dir, f"calib_{stamp}.npz"), known_corners=known_corners, unknown_corners=unknown_corners, T_known2base=T_known2base)
                    print(f"Saved {stamp}" )
                    i+=1
                    ready_to_save = False
                    saved_data.append(
                        {
                            "known_corners": known_corners,
                            "unknown_corners": unknown_corners,
                            "T_known2base": T_known2base,
                        }
                    )

            known_display_img = cv2.resize(known_display_img, (1280, 720), interpolation = cv2.INTER_AREA)
            unknown_display_img = cv2.resize(unknown_display_img, (1280, 720), interpolation = cv2.INTER_AREA)
            cv2.imshow("known_rgb", known_display_img)
            cv2.imshow("unknown_rgb", unknown_display_img)
        # quit on ESC
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            break
            
    cv2.destroyAllWindows()

def get_pose_matrix(x,y,z,roll,target):
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
        T = get_pose_matrix(x, y, z, roll, target)
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
    return x,y,z, roll, target


def main():
    controller_node = ControllerNode()
    known_camera_node = CameraSubscriber("/realsense/left_hand")
    unknown_camera_node = CameraSubscriber("/realsense/head")
    known_intrinsic_path = os.path.join(save_dir, "known_intrinsics.npz")
    unknown_intrinsic_path = os.path.join(save_dir, "unknown_intrinsics.npz")
    unknown_extrinsics_path = os.path.join(save_dir, "unknown_extrinsics.npz")
    vis_thread = threading.Thread(target=vis_and_save, args=(known_camera_node, unknown_camera_node, controller_node, known_intrinsic_path, unknown_intrinsic_path, unknown_extrinsics_path))
    vis_thread.start()
    time.sleep(1)
    print()
    print("camera initialized")


    best_T = None
    target = [1.0, 0.0, 0.2]


    configs =[
        (0.1, 0.4, 0.3, 0.0),
        (0.1, 0.4, 0.0, 0.0),
        (0.1, 0.4, 0.5, 0.0),

        (0.1, 0.2, 0.3, 0.0),
        (0.1, 0.2, 0.0, 0.0),
        (0.1, 0.2, 0.5, 0.0),

        (0.1, 0.0, 0.3, 0.0),
        (0.1, 0.0, 0.0, 0.0),
        (0.1, 0.0, 0.5, 0.0),


    ]


    x = 0.1
    y = 0.4
    z = 0.3
    roll = 0.0
    i = 0
    for x,y,z,roll in configs:
        i+=1
        print(f"\n\n{i+1}")
        collect(x,y,z,roll,target, controller_node)
    print(f"\n\nAll done! Returning home")
    controller_node.go_home(duration=10)

        

if __name__ == "__main__":
    main()