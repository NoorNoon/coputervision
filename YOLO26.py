
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check GPU
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))

# Load YOLO26
model = YOLO("yolo26s.pt")

# Run detection on a sample image
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device=0,
    imgsz=640,
    conf=0.3,
    verbose=True
)

# Save detection result
image = results[0].plot()

output_path = OUTPUT_DIR / "yolo26_test.jpg"

if not cv2.imwrite(str(output_path), image):
    raise RuntimeError("Could not save output image")

print("Detection completed successfully!")
print("Detected objects:", len(results[0].boxes))
print("Saved image:", output_path)

# Display result
cv2.imshow("YOLO26 Detection Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
