import time
import shutil
import numpy as np
import os
import sys
import cv2
file_dir = os.path.dirname(os.path.realpath(__file__))

if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

from controller import ControllerNode
from camerainterface import CameraSubscriber
import threading
from utils import vis_and_save, collect_control_loop, visualize_r_t

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
        [ 0.20, 0.60,  0.20,   0],
        [-0.20, 0.60,  0.20,   0],
        [ 0.20, 0.60, -0.20,  60],
        [-0.20, 0.60, -0.20, -60],

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
def get_config(radius, center):
    configs = []
    cx, cy, cz = center
    steps = 5
    lat_min, lat_max = 90, 270
    lon_min, lon_max = 0, 180
    lat_steps = 5
    lon_steps = 5
    lat_stepsize = (lat_max - lat_min) / lat_steps
    lon_stepsize = (lon_max - lon_min) / lon_steps
    for latitude in np.arange(lat_min+lat_stepsize, lat_max, lat_stepsize):
        for longitutde in np.arange(lon_min+lon_stepsize, lon_max, lon_stepsize):
            lat_rad = np.deg2rad(latitude)
            lon_rad = np.deg2rad(longitutde)
            x = cx + radius * np.sin(lat_rad) * np.cos(lon_rad)
            y = cy + radius * np.cos(lat_rad)
            z = cz + radius * np.sin(lat_rad) * np.sin(lon_rad)
            roll = longitutde  # Vary roll with longitude for diversity
            configs.append([x, y, z, roll])
    return configs
def get_handineye_pose_matrix(x,y,z,roll, target):
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

def collect_handineye_calibration_data(save_dir):
    controller_node = ControllerNode()
    camera_node = CameraSubscriber("/realsense/left_hand")
    vis_thread = threading.Thread(target=vis_and_save, args=(controller_node, [camera_node], "left_wrist_yaw_link", ["left_hand_link"], ["left_hand_color_optical_frame"], (10, 7), save_dir))
    vis_thread.start()
    time.sleep(1)
    print()
    print("camera initialized")

    target_location = [0.1, 0.8, 0.25]
    target = np.array(target_location, dtype=float)

    i = 0
    done = False
    x = 0
    y = 0.3
    z = 0
    roll = 0
    while not done:
        print(f"\n\n{i+1}/{len(configs)} New position: x={x}, y={y}, z={z}, roll={roll}")
        x,y,z,roll, target = collect_control_loop(x,y,z,roll,target, controller_node, get_handineye_pose_matrix, use_right=False)


    print(f"\n\nAll done! Returning home")
    controller_node.go_home(duration=10)            


def main():
    data_dir = os.path.join(file_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    save_dir = os.path.join(data_dir, 'handineye_calibration')


    input(f"press anything to delete {save_dir} and continue")
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    collect_handineye_calibration_data(save_dir)

if __name__ == "__main__":
    main()