import time
import shutil
import os
import sys
import numpy as np
file_dir = os.path.dirname(os.path.realpath(__file__))

if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

from controller import ControllerNode
from camerainterface import CameraSubscriber
import threading

KNOWN_OPTICAL_FRAME = "left_hand_color_optical_frame"
KNOWN_BASE_FRAME = "left_hand_link"
UNKNOWN_OPTICAL_FRAME = "head_color_optical_frame"
UNKNOWN_BASE_FRAME = "head_link"


from collect_handineye_calibration_data import get_handineye_pose_matrix
from utils import vis_and_save, collect_control_loop


def collect_bundle_data(save_dir):
    controller_node = ControllerNode()
    known_camera_node = CameraSubscriber("/realsense/left_hand")
    unknown_camera_node = CameraSubscriber("/realsense/head")
    vis_thread = threading.Thread(target=vis_and_save, args=(controller_node, [known_camera_node, unknown_camera_node], "left_wrist_yaw_link", [KNOWN_BASE_FRAME, UNKNOWN_BASE_FRAME], [KNOWN_OPTICAL_FRAME, UNKNOWN_OPTICAL_FRAME], (10, 7), save_dir))
    vis_thread.start()
    time.sleep(1)
    print()
    print("camera initialized")
    target = np.array([0.8, 0.0, 0.25])
    i = 0
    x = 0.3
    y = 0
    z = 0
    roll = 0
    done = False
    while not done:
        i+=1
        print(f"\n\n{i+1}")
        x,y,z,roll, target = collect_control_loop(x,y,z,roll,target, controller_node, get_handineye_pose_matrix, use_right=False)
    print(f"\n\nAll done! Returning home")
    controller_node.go_home(duration=10)


def main():
    data_dir = os.path.join(file_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    save_dir = os.path.join(data_dir, 'bundle_data')

    input(f"press anything to delete {save_dir} and continue")
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    collect_bundle_data(save_dir)

if __name__ == "__main__":
    main()