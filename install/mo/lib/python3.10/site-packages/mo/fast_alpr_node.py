import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from fast_alpr import ALPR
import cv2
from cv_bridge import CvBridge

class FastALPRNode(Node):
    def __init__(self):
        super().__init__('fast_alpr_node')
        self.alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="global-plates-mobile-vit-v2-model"
        )
        
        
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(String, '/license_plate_data', 10)
        # Explicitly create a named window for display
        cv2.namedWindow("FastALPR - Live Camera Feed", cv2.WINDOW_AUTOSIZE)

    def image_callback(self, msg):
        # Debug log to confirm reception of an image
        self.get_logger().info("Received an image frame")

        try:
            # Convert ROS Image message to OpenCV image format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        # Run ALPR on the frame
        results = self.alpr.predict(frame)

        # Annotate the frame with predictions
        annotated_frame = self.alpr.draw_predictions(frame)

        # Publish detected license plate data (if any)
        if results:
            plate_data = results[0]['text']
            self.publisher.publish(String(data=plate_data))
            self.get_logger().info(f"Detected license plate: {plate_data}")

        # Display the annotated frame in a GUI window
        cv2.imshow("FastALPR - Live Camera Feed", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = FastALPRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down FastALPR node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

