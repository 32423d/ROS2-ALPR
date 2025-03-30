#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import fast_alpr  # Import your Fast-ALPR library

class FastALPRSubscriber(Node):
    def __init__(self):
        super().__init__('fast_alpr_subscriber')
        # Subscribe to the camera feed published on 'camera_feed'
        self.subscription = self.create_subscription(
            Image,
            'camera_feed',
            self.listener_callback,
            10)
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        # Convert the ROS Image message to an OpenCV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        # ALPR
        try:
            # Initialize ALPR
            alpr = ALPR(
                detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr_model="global-plates-mobile-vit-v2-model",
            )

            # Run ALPR prediction on the current frame
            results = alpr.predict(cv_image)

            # Debug Functions 
            '''
            # Draw predictions on the frame
            annotated_frame = alpr.draw_predictions(cv_image)

            # Display the annotated frame
            cv2.imshow("FastALPR - Live Camera Feed", annotated_frame)
            '''

            # Log Results
            self.get_logger().info(f"ALPR Results: {results}")


        except Exception as e:
            self.get_logger().error(f"Fast-ALPR processing failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = FastALPRSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down ALPR subscriber node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()