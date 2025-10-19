import itertools
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

from utils import vis_and_save, collect_control_loop


def get_handtoeye_pose_matrix(x,y,z,roll,target):
    roll += 180
    pos = np.array([x, y, z], dtype=float)
    dir_vec = target - pos
    norm = np.linalg.norm(dir_vec)
    x_axis = -1 * (dir_vec / norm)

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

def collect_handtoeye_calibration_data(save_dir):
    controller_node = ControllerNode()
    camera_node = CameraSubscriber("/realsense/head")
    vis_thread = threading.Thread(target=vis_and_save, args=(controller_node, [camera_node], "right_wrist_yaw_link", ["head_link"], ["head_color_optical_frame"], (10, 7), save_dir))
    vis_thread.start()
    time.sleep(1)
    print()
    print("camera initialized")


    target = np.array([0.0, 0.0, 0.68])
    x = 0.5
    y = 0
    z = 0
    roll = 0
    done = False
    i=0
    while not done:
        print(f"\n\n{i+1}")
        i+=1
        x,y,z,roll,target = collect_control_loop(x, y, z, roll, target, controller_node, get_handtoeye_pose_matrix, use_right=True)
        inp = input("press y to exit: ")
        if inp == "y":
            done = True
    print(f"\n\nAll done! Returning home")
    controller_node.go_home(duration=10)
        

def main():
    data_dir = os.path.join(file_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    save_dir = os.path.join(data_dir, 'handtoeye_calibration')


    input(f"press anything to delete {save_dir} and continue")
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    collect_handtoeye_calibration_data(save_dir)

if __name__ == '__main__':
    main()