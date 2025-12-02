import cv2
import os
import time
from ultralytics import YOLO

WINDOW_NAME = "c.linjewile Face Detection built on YOLO"
MODEL_PATH = "yolov8n.pt"   # you can swap for 'yolov8s.pt' if you want


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    seconds = max(0, int(seconds))
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"


def process_frame(model, frame):
    """
    Run YOLO on a single frame and return:
    - annotated_frame (with boxes + labels)
    - person_count
    """
    results = model(frame, verbose=False)
    annotated_frame = frame.copy()
    person_count = 0
    
    # Get frame dimensions for scaling
    frame_height, frame_width = frame.shape[:2]
    font_scale = max(0.4, min(0.8, frame_width / 1000))  # Scale font with width
    thickness = max(1, int(frame_width / 500))  # Scale thickness with width

    # YOLO can return multiple results, but for webcam/video it's usually 1 per frame
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

            # Draw rectangle and label - scaled to frame size
            label = f"person {conf * 100:.1f}%"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                (0, 255, 0),
                thickness
            )

    # Display person count on the frame - scaled to frame size
    count_text = f"People Count: {person_count}"
    count_y_position = int(frame_height * 0.07)  # 7% from top
    cv2.putText(
        annotated_frame,
        count_text,
        (10, count_y_position),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale * 1.2,
        (0, 0, 255),
        thickness
    )

    return annotated_frame, person_count


def run_webcam(model):
    """Webcam mode (very close to your original main loop)."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Optional: set a smaller resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Webcam mode. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        annotated_frame, _ = process_frame(model, frame)

        # Show the result
        cv2.imshow(f"{WINDOW_NAME} (press 'q' to quit)", annotated_frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_video_file(model, video_path):
    """Video-file mode with progress info and pause-on-last-frame."""
    if not os.path.exists(video_path):
        print(f"Error: File not found -> {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file -> {video_path}")
        return

    # Get original video dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video resolution: {frame_width}x{frame_height}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / video_fps if (video_fps and total_frames > 0) else None

    # Ask user if they want to save the output
    save_choice = input("Do you want to save the processed video? (y/n): ").strip().lower()
    save_video = save_choice == 'y'

    out = None
    output_filename = None
    if save_video:
        # Setup output video writer
        output_filename = f"detected_{os.path.basename(video_path)}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, video_fps, (frame_width, frame_height))
        print(f"[INFO] Output will be saved as: {output_filename}")

    print(f"[INFO] Video mode. File: {video_path}")
    if video_fps and total_frames > 0:
        print(f"[INFO] Total frames: {total_frames}, FPS: {video_fps:.2f}, Duration: {format_time(duration_sec)}")
    else:
        print("[INFO] FPS or frame count unknown.")
    print("[INFO] Press 'q' to quit early.")

    frame_idx = 0
    last_frame = None
    natural_end = False

    while True:
        ret, frame = cap.read()
        if not ret:
            # Reached the end of the video
            natural_end = True
            break

        frame_idx += 1
        last_frame = frame.copy()

        annotated_frame, _ = process_frame(model, frame)

        # Progress info - adjust position based on frame height
        if duration_sec and video_fps:
            current_time_in_video = frame_idx / video_fps
            progress_pct = (frame_idx / total_frames) * 100.0
            time_text = f"{format_time(current_time_in_video)} / {format_time(duration_sec)}"
            progress_text = f"{progress_pct:5.1f}%"
        else:
            time_text = "Time: --:-- / --:--"
            progress_text = ""

        # Draw progress info - position scales with frame size
        bottom_text = f"{time_text}   {progress_text}"
        progress_y_position = int(frame_height * 0.15)  # 15% from top
        font_scale = max(0.4, min(0.8, frame_width / 1000))  # Scale font with width
        
        cv2.putText(
            annotated_frame,
            bottom_text,
            (10, progress_y_position),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (0, 0, 255),
            1
        )

        cv2.imshow(WINDOW_NAME, annotated_frame)

        # Write the annotated frame to output video if saving
        if save_video and out is not None:
            out.write(annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # User quit early
            natural_end = False
            break

    cap.release()
    if out is not None:
        out.release()

    if save_video and output_filename:
        print(f"[INFO] Video saved as: {output_filename}")

    # If we hit the natural end of the video, pause on the last frame with a message
    if natural_end and last_frame is not None:
        finished_frame, _ = process_frame(model, last_frame)
        message = "Finished processing video. Press 'q' to close."
        cv2.putText(
            finished_frame,
            message,
            (30, 40),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        while True:
            cv2.imshow(WINDOW_NAME, finished_frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyAllWindows()


def main():
    # Load YOLO model
    print("[INFO] Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    # Prompt user for source
    print("Select input source:")
    print("1 - Webcam")
    print("2 - Video file")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        run_webcam(model)
    elif choice == "2":
        video_path = input("Enter path to video file: ").strip().strip('"')
        run_video_file(model, video_path)
    else:
        print("Invalid choice. Please run again and enter 1 or 2.")


if __name__ == "__main__":
    main()
