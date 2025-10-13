import threading
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage


class CameraSubscriber(Node):
    def __init__(self, camera_name_space):
        print("Initializing CameraSubscriber for camera:", camera_name_space)
        node_name = f"{camera_name_space.replace(' ', '_').replace('/', '')}_subscriber"
        super().__init__(node_name)
        self.camera_name_space = camera_name_space
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_info = None
        self._lock = threading.Lock()

        # QoS for sensor data (Best effort + TRANSIENT_LOCAL)
        sensor_data_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriptions
        self.create_subscription(
            CompressedImage,
            f"{camera_name_space}/color/image_raw/compressed",
            self._rgb_callback,
            sensor_data_qos,
        )
        self.create_subscription(
            CameraInfo,
            f"{camera_name_space}/color/camera_info",
            self._info_callback,
            sensor_data_qos,
        )

        # Private executor & spin thread for this node
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True
        )
        self._spin_thread.start()

    def _rgb_callback(self, msg: CompressedImage):
        try:
            rgb_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            with self._lock:
                # print("[REALSENSE SUB rgb] lock aquired")
                self.latest_rgb = rgb_img
                # print("[REALSENSE SUB rgb] lock released")

        except Exception as e:
            print(f"Error processing RGB image for {self.camera_name}: {e}")


    def _info_callback(self, msg: CameraInfo):
        with self._lock:
            # print("[REALSENSE SUB info] lock aquired")
            self.latest_info = msg
            # print("[REALSENSE SUB info] lock released")

    def get_data(self):
        rgb, info = None, None
        if self.latest_rgb is not None and self.latest_info is not None:
            with self._lock:
                # print("[REALSENSE SUB get data] lock aquired")

                rgb = self.latest_rgb
                info = self.latest_info

                # Clear buffers
                self.latest_rgb = None
                self.latest_info = None
                # print("[REALSENSE SUB get data] lock released")

        return rgb, info

    def shutdown(self):
        # Stop spinning and clean up
        self._executor.shutdown()
        self._spin_thread.join(timeout=1.0)
        self.destroy_node()

    def __str__(self):
        return f"RealSenseSubscriber(camera_name={self.camera_name})"

    def __repr__(self):
        return self.__str__()