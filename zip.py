
from pathlib import Path
from zipfile import ZipFile, BadZipFile

# Folder containing this Python file
DATA_DIR = Path(__file__).resolve().parent

print("Dataset folder:", DATA_DIR)

for filename in ["train.zip", "test.zip"]:

    zip_path = DATA_DIR / filename

    print("\n" + "=" * 50)
    print("Checking:", filename)

    if not zip_path.exists():
        print("File not found:", zip_path)
        continue

    size_gb = zip_path.stat().st_size / (1024 ** 3)

    print(f"Archive size: {size_gb:.2f} GB")

    try:
        with ZipFile(zip_path, "r") as archive:

            files = archive.namelist()

            print("Total entries:", len(files))

            print("\nFirst 30 entries:")

            for name in files[:30]:
                print(name)

            images = [
                f for f in files
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]

            annotations = [
                f for f in files
                if f.lower().endswith(
                    (".txt", ".json", ".xml")
                )
            ]

            print("\nImage files:", len(images))
            print("Annotation files:", len(annotations))

    except BadZipFile:
        print("ERROR: Invalid or incomplete ZIP file")

print("\nInspection completed!")
