import cv2
from object_detector import ObjectDetector
from distance_estimator import DistanceEstimator

def main():
    # === Calibration constants ===
    KNOWN_WIDTH_CM = 5.0     # width of your reference object in cm
    FOCAL_LENGTH = 800.0     # compute via calibration step

    # initialize detector and estimator
    detector = ObjectDetector()
    estimator = DistanceEstimator(focal_length=FOCAL_LENGTH,
                                  known_width_cm=KNOWN_WIDTH_CM)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        for det in detections:
            x1, y1, x2, y2 = det['box']
            pixel_w = x2 - x1
            distance = estimator.estimate(pixel_w)
            label = f"{det['class']} {distance:.2f} cm" if distance else det['class']

            # draw bounding box + distance
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Distance Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
