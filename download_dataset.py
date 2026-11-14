
from pathlib import Path
from huggingface_hub import hf_hub_download

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset folder
DATA_DIR = BASE_DIR / "data" / "downloads"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# SoccerNet dataset
REPO_ID = "SoccerNet/SN-Tracking-2023"

print("Starting SoccerNet dataset download...")
print("File: train.zip")
print("Size: approximately 9.58 GB")

try:
    file_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="train.zip",
        repo_type="dataset",
        local_dir=str(DATA_DIR)
    )

    print("\nDownload completed!")
    print("Saved to:", file_path)

except Exception as error:
    print("\nDownload failed:")
    print(error)
