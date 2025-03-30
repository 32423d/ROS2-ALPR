#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        # Create a publisher for the camera feed on topic 'camera_feed'
        self.publisher_ = self.create_publisher(Image, 'camera_feed', 10)
        # Publish at a rate of 10 Hz
        self.timer = self.create_timer(0.1, self.timer_callback)
        # Open the default camera (device index 0)
        self.cap = cv2.VideoCapture(0)
        self.bridge = CvBridge()

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the OpenCV image (BGR) to a ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher_.publish(ros_image)
            self.get_logger().info('Published a camera frame')
        else:
            self.get_logger().error('Failed to capture frame from camera')

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down camera publisher node...')
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()