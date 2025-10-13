import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from custom_ros_messages.srv import Query, UpdateTrackedObject, UpdateBeliefs, ResetBeliefs
from custom_ros_messages.action import DualArm
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R
import time
from unitree_go.msg import MotorCmds, MotorCmd
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from rclpy.executors import SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading



def pose_to_matrix(pose_array):
    x, y, z, roll, pitch, yaw = pose_array
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    matrix = np.eye(4)
    matrix[:3, :3] = r.as_matrix()
    matrix[:3, 3] = [x, y, z]
    return matrix

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
        lM[:3, 3] = [0.3, 0.5, 0.2]
        rM = np.eye(4)
        rM[:3, 3] = [0.3, -0.5, 0.2]
        self.r_arm_mat = rM
        self.l_arm_mat = lM
        self.marker_pub = self.create_publisher(Marker, "/camera_marker", 10)

        self.hand_pub = self.create_publisher(MotorCmds, '/inspire/cmd', 10)
        self.hand_length = 0.3
        
        self.l_hand = None
        self.r_hand = None

        PC_QOS = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pointcloud_pub = self.create_publisher(PointCloud2, "/experiment_pointcloud", PC_QOS)



        self._tf_cb_group = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_node = Node('tf_helper', callback_group=self._tf_cb_group)
        self.tf_listener = TransformListener(self.tf_buffer, self.tf_node, callback_group=self._tf_cb_group)

        self._tf_executor = SingleThreadedExecutor()
        self._tf_executor.add_node(self.tf_node)
        self._tf_thread = threading.Thread(target=self._tf_executor.spin, daemon=True)
        self._tf_thread.start()

        self.go_home()
        self.open_hands()
        time.sleep(1)
        self.close_hands()
        time.sleep(1)


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

        raise RuntimeError(f"TF {source_frame} -> {target_frame} not available within {timeout:.2f}s: {last_exc}")


    def publish_marker(self, x,y,z):
        marker = Marker()

        marker.header.frame_id = "pelvis"
        marker.ns = "behavior marker"

        marker.type = Marker.DELETEALL
        self.marker_pub.publish(marker)



        marker.type = Marker.SPHERE
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)


    def set_hands(self, l_goal=None, r_goal=None):
        if l_goal is not None:
            self.l_hand = l_goal
        if r_goal is not None:
            self.r_hand = r_goal

        msg = MotorCmds()
        msg.cmds = [MotorCmd(mode=1, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0) for i in range(12)]
        for i, (l_q, r_q) in enumerate(zip(self.l_hand, self.r_hand)):
            msg.cmds[i].q = r_q
            msg.cmds[6+i].q = l_q

        self.hand_pub.publish(msg)

    def open_hands(self):
        l_hand = [1.0]*6
        r_hand = [1.0]*6
        self.set_hands(l_goal = l_hand, r_goal = r_hand)
    def close_hands(self):
        l_hand = [0.0]*6
        l_hand[5] = 1.0
        r_hand = [0.0]*6
        r_hand[5] = 1.0
        self.set_hands(l_goal = l_hand, r_goal = r_hand)

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
        # print(f'\rLeft Error Linear: {feedback.left_error_linear:.2f} Angular: {feedback.left_error_angular:.2f}; T:{time.time()- self.start_time:.2f}', end="", flush=True)
        print(f'\rRight Error Linear: {feedback.right_error_linear:.2f} Angular: {feedback.right_error_linear:.2f} T:{time.time()- self.start_time:.2f}', end="", flush=True)

    def close(self):
        self.destroy_node()
        rclpy.shutdown()