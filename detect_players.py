
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# 1. Project folders
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "data" / "match.mp4"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "tracked_match.mp4"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Check input video
if not VIDEO_PATH.is_file():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}\n"
        "Put match.mp4 inside the data folder."
    )


# 3. Select GPU or CPU
device = 0 if torch.cuda.is_available() else "cpu"

print("Using device:", device)

if device == 0:
    print("GPU:", torch.cuda.get_device_name(0))


# 4. Load YOLO
model = YOLO("yolo11n.pt")


# 5. Open video
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open the video.")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("Invalid video dimensions.")


# 6. Create output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")


# 7. Process video frames
frame_number = 0
unique_ids = set()

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # Detect and track people
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.30,
            iou=0.50,
            device=device,
            verbose=False
        )

        result = results[0]

        # Draw bounding boxes and tracking IDs
        annotated_frame = result.plot()

        current_count = len(result.boxes)

        # Collect tracking IDs
        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):
            ids = result.boxes.id.int().cpu().tolist()
            unique_ids.update(ids)

        # Display information
        cv2.putText(
            annotated_frame,
            f"Detected people: {current_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        writer.write(annotated_frame)

        cv2.imshow(
            "Football Player Tracking",
            annotated_frame
        )

        if frame_number % 30 == 0:
            print(f"Processed {frame_number} frames")

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# 8. Final summary
print("\nProcessing completed!")
print("Total processed frames:", frame_number)
print("Unique tracking IDs:", len(unique_ids))
print("Output saved to:", OUTPUT_PATH)
