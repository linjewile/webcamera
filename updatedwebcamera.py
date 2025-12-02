import cv2
from ultralytics import YOLO

def main():
    # Load a pretrained YOLO model (you can use 'yolov8n.pt', 'yolov8s.pt', etc.)
    # 'n' = nano (fast), 's' = small (better accuracy, a bit slower)
    model = YOLO("yolov8n.pt")  # this will auto-download the weights the first time

    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Optional: set a smaller resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Run YOLO on the frame
        results = model(frame, verbose=False)

        # YOLO can return multiple results, but for webcam it's usually 1 per frame
        annotated_frame = frame.copy()
        person_count = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])      # class id
                conf = float(box.conf[0])     # confidence
                # Ultralytics' default COCO labels: class 0 is 'person'
                if cls_id != 0:
                    continue  # skip non-person

                person_count += 1

                # Get bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                # Draw rectangle and label
                label = f"person {conf*100:.1f}%"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (0, 255, 0),
                    1
                )

        # Display person count on the frame
        count_text = f"People Count: {person_count}"
        cv2.putText(
            annotated_frame,
            count_text,
            (10, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # Show the result
        cv2.imshow("c.linjewile face Detection built on YOLO(press 'q' to quit)", annotated_frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
