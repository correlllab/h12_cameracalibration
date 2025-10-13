import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from custom_ros_messages.action import DualArm
from scipy.spatial.transform import Rotation as R
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
        self.action_client = ActionClient(
            self,
            DualArm,
            'move_dual_arm'
        )
        lM = np.eye(4)
        lM[:3, 3] = [0.3, 0.3, 0.2]
        rM = np.eye(4)
        rM[:3, 3] = [0.3, -0.3, 0.2]
        self.r_arm_mat = rM
        self.l_arm_mat = lM
        
        self.l_hand = None
        self.r_hand = None


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
        self.go_home()


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
                T = np.eye(4, dtype=float)
                T[:3, :3] = Rm
                T[:3,  3] = [t.x, t.y, t.z]
                return T

            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                last_exc = e
                time.sleep(0.02)  # brief backoff

        #raise RuntimeError(f"TF {source_frame} -> {target_frame} not available within {timeout:.2f}s: {last_exc}")
        return None


    def go_home(self, duration=10):
        goal_msg = DualArm.Goal()
        goal_msg.duration = duration
        goal_msg.keyword = "home"

        self.action_client.wait_for_server()

        # send action
        print('Going home...')
        self.start_time = time.time()
        future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected')
            return

        future_result = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, future_result)
        result = future_result.result().result
        print()
        print(f'Final result: success = {result.success}')
        time.sleep(1)

    def send_arm_goal(self, left_mat = None, right_mat = None, duration=3, block = True):
        assert left_mat is None or left_mat.shape == (4,4)
        assert right_mat is None or right_mat.shape == (4,4)
        assert duration > 0 and isinstance(duration, int)
        if left_mat is not None:
            self.l_arm_mat = left_mat
        if right_mat is not None:
            self.r_arm_mat = right_mat

        goal_msg = DualArm.Goal()
        goal_msg.left_target = self.l_arm_mat.reshape(-1).tolist()
        goal_msg.right_target = self.r_arm_mat.reshape(-1).tolist()
        goal_msg.duration = duration

        self.action_client.wait_for_server()

        # send action
        print('Sending goal...')
        self.start_time = time.time()
        future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected')
            return

        # start a cancel listener thread
        # print('Goal accepted, waiting for result...')

        if block:
            # wait till finish
            future_result = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, future_result)
            result = future_result.result().result
            time.sleep(1)
        print()
        print(f'Final result: success = {result.success}')


    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # print(f'\rLeft Error Linear: {feedback.left_error_linear:.2f} Angular: {feedback.left_error_angular:.2f}; Right Error Linear: {feedback.right_error_linear:.2f} Angular: {feedback.right_error_linear:.2f} T:{time.time()- self.start_time:.2f}', end="", flush=True)
        print(f'\rLeft Error Linear: {feedback.left_error_linear:.2f} Angular: {feedback.left_error_angular:.2f}; T:{time.time()- self.start_time:.2f}', end="", flush=True)
        # print(f'\rRight Error Linear: {feedback.right_error_linear:.2f} Angular: {feedback.right_error_linear:.2f} T:{time.time()- self.start_time:.2f}', end="", flush=True)

    def close(self):
        self.destroy_node()
        rclpy.shutdown()