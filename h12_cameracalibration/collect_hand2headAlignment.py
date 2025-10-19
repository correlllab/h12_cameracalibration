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
KNOWN_BASE_FRAME = "left_hand_link"
UNKNOWN_OPTICAL_FRAME = "head_color_optical_frame"
UNKNOWN_BASE_FRAME = "head_link"
BASEFRAME = "pelvis"


from collect_handineye_calibration_data import get_handineye_pose_matrix
from utils import vis_and_save, collect_control_loop

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


def main():
    controller_node = ControllerNode()
    known_camera_node = CameraSubscriber("/realsense/left_hand")
    unknown_camera_node = CameraSubscriber("/realsense/head")
    vis_thread = threading.Thread(target=vis_and_save, args=(known_camera_node, unknown_camera_node, controller_node, known_intrinsic_path, unknown_intrinsic_path, unknown_extrinsics_path))
    vis_thread.start()
    time.sleep(1)
    print()
    print("camera initialized")
    target = [1.0, 0.0, 0.2]
    for x,y,z,roll in configs:
        i+=1
        print(f"\n\n{i+1}")
        collect_control_loop(x,y,z,roll,target, controller_node, get_handineye_pose_matrix, right_arm=False)
    print(f"\n\nAll done! Returning home")
    controller_node.go_home(duration=10)

        

if __name__ == "__main__":
    main()