
from pathlib import Path
from zipfile import ZipFile

BASE_DIR = Path(__file__).resolve().parent

ZIP_PATH = BASE_DIR / "data" / "downloads" / "train.zip"

OUTPUT_DIR = BASE_DIR / "data" / "extracted"

SEQUENCE = "SNMOT-060"
PREFIX = f"train/{SEQUENCE}/"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {ZIP_PATH}"
    )

print("Extracting sequence:", SEQUENCE)

with ZipFile(ZIP_PATH, "r") as archive:

    files = [
        file for file in archive.namelist()
        if file.startswith(PREFIX)
    ]

    print("Files to extract:", len(files))

    for index, file in enumerate(files, start=1):
        archive.extract(file, OUTPUT_DIR)

        if index % 100 == 0:
            print(f"Extracted {index}/{len(files)}")

print("\nExtraction completed!")
print("Saved to:", OUTPUT_DIR / "train" / SEQUENCE)
