import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
# from scipy.spatial.transform import Rotation as R
import time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from rclpy.executors import SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading



class ControllerNode(Node):
    def __init__(self):
        rclpy.init()
        super().__init__('controller')
        
    
        self._tf_cb_group = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_node = Node('tf_helper')
        self.tf_listener = TransformListener(self.tf_buffer, self.tf_node)

        self._tf_executor = SingleThreadedExecutor()
        self._tf_executor.add_node(self.tf_node)
        self._tf_thread = threading.Thread(target=self._tf_executor.spin, daemon=True)
        print("Starting TF thread")
        self._tf_thread.start()
        print("TF thread started")



    def get_tf(self, source_frame: str, target_frame: str, timeout: float = 1.0):
        """
        Look up the transform from `source_frame` --> `target_frame`.

        Args:
            source_frame: frame you HAVE (e.g., 'camera_link')
            target_frame: frame you WANT (e.g., 'base_link')
            timeout: seconds to wait/retry for the TF to become available
            as_matrix: if True, return a 4x4 numpy homogeneous transform; otherwise return TransformStamped
        Returns:
            TransformStamped or 4x4 numpy array
        Raises:
            RuntimeError if not available within timeout
        """
        deadline = time.time() + timeout
        last_exc = None
        while time.time() < deadline and rclpy.ok():
            try:
                # Empty Time() = "latest available"
                ts = self.tf_buffer.lookup_transform(
                    target_frame,                  # target
                    source_frame,                  # source
                    rclpy.time.Time())            # latest

                # Convert to 4x4
                t = ts.transform.translation
                q = ts.transform.rotation
                Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                H = np.eye(4, dtype=float)
                H[:3, :3] = Rm
                H[:3,  3] = [t.x, t.y, t.z]
                return H

            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                last_exc = e
                time.sleep(0.02)  # brief backoff

        #raise RuntimeError(f"TF {source_frame} -> {target_frame} not available within {timeout:.2f}s: {last_exc}")
        return None
    
    # def get_frame_in_pelvis(self, frame_name: str):
    #     self.robotmodel.update_kinematics()
    #     tf = self.robotmodel.get_frame_transformation(frame_name)
    #     return tf


    def go_home(self, duration=10):
        return None

    def send_arm_goal(self, left_mat = None, right_mat = None, duration=3, block = True):
        return None


    def close(self):
        self.destroy_node()
        rclpy.shutdown()

def main():
    controller = ControllerNode()
    controller.go_home()
    controller.close()