import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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
        people_count = 0  # Counter for people in current frame

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])      # class id
                conf = float(box.conf[0])     # confidence
                # Ultralytics' default COCO labels: class 0 is 'person'
                if cls_id != 0:
                    continue  # skip non-person

                people_count += 1  # Increment people counter

                # Get bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                # Draw rectangle and label (red box)
                label = f"person {conf*100:.1f}%"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

        # Display people count on frame using Times New Roman
        count_text = f"People Count: {people_count}"
        
        # Convert to PIL Image to use TrueType fonts
        pil_img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Try to use Times New Roman, fallback to default if not available
        try:
            font = ImageFont.truetype("times.ttf", 40)  # Times New Roman on Windows
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 40)
            except:
                font = ImageFont.load_default()
        
        # Draw text with black color (RGB format for PIL)
        draw.text((10, 10), count_text, font=font, fill=(0, 0, 0))
        
        # Convert back to OpenCV format
        annotated_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Show the result
        cv2.imshow("C.Linjewile's people detection built using YOLO (press 'q' to quit)", annotated_frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
