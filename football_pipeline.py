
from pathlib import Path
from collections import defaultdict
import csv
import configparser

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO


# ==================================================
# PART 1: PROJECT CONFIGURATION
# Define the dataset, model, and experiment settings.
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

SPLIT = "train"
SEQUENCE = "SNMOT-060"

SEQ_DIR = BASE_DIR / "data" / "extracted" / SPLIT / SEQUENCE
IMAGE_DIR = SEQ_DIR / "img1"

OUTPUT_DIR = BASE_DIR / "outputs" / SEQUENCE / "v2"
DEBUG_DIR = OUTPUT_DIR / "debug"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolo26s.pt"
CONFIDENCE = 0.30

MAX_FRAMES = 150
MIN_TRACK_FRAMES = 3

# Number of color groups, including possible referees.
N_CLUSTERS = 3

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)
print("Sequence:", SEQUENCE)


# ==================================================
# PART 2: LOAD SOCCERNET DATASET
# Read consecutive frames and sequence metadata.
# ==================================================

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset folder not found: {IMAGE_DIR}"
    )

images = sorted(IMAGE_DIR.glob("*.jpg"))

if not images:
    raise RuntimeError("No image frames found")

images = images[:MAX_FRAMES]

config = configparser.ConfigParser()
config.read(
    SEQ_DIR / "seqinfo.ini",
    encoding="utf-8-sig"
)

fps = config.getfloat(
    "Sequence",
    "frameRate",
    fallback=25
)

if fps <= 0:
    fps = 25

first_frame = cv2.imread(str(images[0]))

if first_frame is None:
    raise RuntimeError("Cannot read first frame")

height, width = first_frame.shape[:2]

print("Frames:", len(images))
print("Resolution:", width, "x", height)
print("FPS:", fps)


# ==================================================
# PART 3: LOAD YOLO26
# Detect people and track them using ByteTrack.
# ==================================================

model = YOLO(MODEL_NAME)

print("YOLO26 loaded successfully")


# ==================================================
# PART 4: EXTRACT JERSEY REGION
# Crop the central upper-body area of each person.
# ==================================================

def extract_jersey(frame, box):

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    w = x2 - x1
    h = y2 - y1

    if w < 10 or h < 20:
        return None

    # Keep the central torso to reduce background.
    left = x1 + int(w * 0.25)
    right = x1 + int(w * 0.75)

    top = y1 + int(h * 0.18)
    bottom = y1 + int(h * 0.48)

    jersey = frame[top:bottom, left:right]

    if jersey.size == 0:
        return None

    return jersey


# ==================================================
# PART 5: RGB BASELINE
# Extract raw RGB color statistics without masking.
# ==================================================

def extract_rgb_features(jersey):

    rgb = cv2.cvtColor(
        jersey,
        cv2.COLOR_BGR2RGB
    )

    pixels = rgb.reshape(-1, 3).astype(float)

    median = np.median(pixels, axis=0)
    mean = np.mean(pixels, axis=0)

    return np.concatenate([median, mean])


# ==================================================
# PART 6: HSV PREPROCESSING AND SEGMENTATION
# Remove pitch-like green pixels and apply morphology.
# ==================================================

def extract_hsv_features(jersey):

    hsv = cv2.cvtColor(
        jersey,
        cv2.COLOR_BGR2HSV
    )

    # Approximate grass color range.
    lower_green = np.array([35, 35, 25])
    upper_green = np.array([90, 255, 255])

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    mask = cv2.bitwise_not(green_mask)

    # Remove noise and fill small holes.
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    pixels = hsv[mask > 0]

    if len(pixels) < 10:
        return None, mask

    hue = pixels[:, 0].astype(float)
    saturation = pixels[:, 1].astype(float) / 255.0
    value = pixels[:, 2].astype(float) / 255.0

    angle = hue * (2 * np.pi / 180)

    features = np.array([
        np.mean(np.cos(angle) * saturation),
        np.mean(np.sin(angle) * saturation),
        np.median(saturation),
        np.median(value),
        np.mean(saturation),
        np.mean(value)
    ])

    return features, mask


# ==================================================
# PART 7: DETECTION, TRACKING, AND FEATURE EXTRACTION
# Process all frames and store player observations.
# ==================================================

records = []
debug_saved = 0

for index, image_path in enumerate(images, start=1):

    frame = cv2.imread(str(image_path))

    if frame is None:
        print("Skipping unreadable frame:", image_path.name)
        continue

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is None or boxes.id is None:
        continue

    coordinates = boxes.xyxy.cpu().numpy()
    track_ids = boxes.id.int().cpu().tolist()
    confidences = boxes.conf.cpu().numpy()

    for box, track_id, confidence in zip(
        coordinates,
        track_ids,
        confidences
    ):

        jersey = extract_jersey(frame, box)

        if jersey is None:
            continue

        rgb = extract_rgb_features(jersey)
        hsv, mask = extract_hsv_features(jersey)

        # Save some jersey crops and segmentation masks.
        if debug_saved < 30:

            name = f"frame{index}_id{track_id}"

            cv2.imwrite(
                str(DEBUG_DIR / f"{name}_jersey.jpg"),
                jersey
            )

            cv2.imwrite(
                str(DEBUG_DIR / f"{name}_mask.png"),
                mask
            )

            debug_saved += 1

        records.append({
            "frame": int(image_path.stem),
            "track_id": int(track_id),
            "box": box.tolist(),
            "confidence": float(confidence),
            "rgb": rgb,
            "hsv": hsv
        })

    if index % 25 == 0:
        print(f"Processed {index}/{len(images)} frames")


# ==================================================
# PART 8: TEMPORAL FEATURE AGGREGATION
# Aggregate observations for each tracking identity.
# ==================================================

tracks = defaultdict(list)

for record in records:
    tracks[record["track_id"]].append(record)

track_ids = []
rgb_data = []
hsv_data = []

for track_id, observations in tracks.items():

    valid = [
        item for item in observations
        if item["hsv"] is not None
    ]

    if len(valid) < MIN_TRACK_FRAMES:
        continue

    track_ids.append(track_id)

    rgb_data.append(
        np.median(
            [item["rgb"] for item in valid],
            axis=0
        )
    )

    hsv_data.append(
        np.median(
            [item["hsv"] for item in valid],
            axis=0
        )
    )

if len(track_ids) < N_CLUSTERS:
    raise RuntimeError(
        "Not enough valid tracks for clustering"
    )

rgb_data = np.asarray(rgb_data)
hsv_data = np.asarray(hsv_data)


# ==================================================
# PART 9: RGB AND HSV CLUSTERING
# Standardize features and cluster independently.
# ==================================================

rgb_scaler = StandardScaler()
hsv_scaler = StandardScaler()

rgb_scaled = rgb_scaler.fit_transform(rgb_data)
hsv_scaled = hsv_scaler.fit_transform(hsv_data)

rgb_model = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

hsv_model = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

rgb_labels = rgb_model.fit_predict(rgb_scaled)
hsv_labels = hsv_model.fit_predict(hsv_scaled)

rgb_clusters = dict(zip(track_ids, rgb_labels))
hsv_clusters = dict(zip(track_ids, hsv_labels))

print("RGB and HSV clustering completed")


# ==================================================
# PART 10: EXPORT CSV RESULTS
# Store detections and both clustering predictions.
# ==================================================

csv_path = OUTPUT_DIR / "player_results_v2.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "frame",
        "track_id",
        "confidence",
        "x1", "y1", "x2", "y2",
        "rgb_cluster",
        "hsv_cluster",
        "rgb_features",
        "hsv_features"
    ])

    for record in records:

        track_id = record["track_id"]

        writer.writerow([
            record["frame"],
            track_id,
            record["confidence"],
            *record["box"],
            rgb_clusters.get(track_id, -1),
            hsv_clusters.get(track_id, -1),
            record["rgb"].tolist(),
            (
                record["hsv"].tolist()
                if record["hsv"] is not None
                else None
            )
        ])


# ==================================================
# PART 11: VIDEO VISUALIZATION
# Generate separate RGB and HSV clustering videos.
# ==================================================

def create_video(filename, cluster_map):

    output_path = OUTPUT_DIR / filename

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot create video: {output_path}"
        )

    frame_records = defaultdict(list)

    for record in records:
        frame_records[record["frame"]].append(record)

    colors = {
        0: (255, 100, 0),
        1: (0, 0, 255),
        2: (0, 255, 255)
    }

    try:
        for image_path in images:

            frame = cv2.imread(str(image_path))

            if frame is None:
                continue

            frame_id = int(image_path.stem)

            for record in frame_records[frame_id]:

                track_id = record["track_id"]
                cluster = cluster_map.get(track_id, -1)

                x1, y1, x2, y2 = map(
                    int, record["box"]
                )

                color = colors.get(
                    cluster,
                    (180, 180, 180)
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label = (
                    f"ID {track_id} C{cluster}"
                    if cluster >= 0
                    else f"ID {track_id} Unknown"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

            writer.write(frame)

    finally:
        writer.release()

    print("Video saved:", output_path)


create_video(
    "rgb_classification.mp4",
    rgb_clusters
)

create_video(
    "hsv_classification.mp4",
    hsv_clusters
)


# ==================================================
# PART 12: FINAL SUMMARY
# Display the generated experiment outputs.
# ==================================================

print("\nPIPELINE V2 COMPLETED")
print("Sequence:", SEQUENCE)
print("Frames:", len(images))
print("Observations:", len(records))
print("Tracked identities:", len(tracks))
print("Classified tracks:", len(track_ids))
print("CSV:", csv_path)
print("Output folder:", OUTPUT_DIR)
