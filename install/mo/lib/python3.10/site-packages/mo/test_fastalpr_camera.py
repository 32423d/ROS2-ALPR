import cv2
from fast_alpr import ALPR

# Initialize ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

# Open camera feed (use 0 for default webcam or replace with IP camera URL)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run ALPR prediction on the current frame
    results = alpr.predict(frame)

    # Draw predictions on the frame
    annotated_frame = alpr.draw_predictions(frame)

    # Display the annotated frame
    cv2.imshow("FastALPR - Live Camera Feed", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

