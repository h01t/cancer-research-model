import csv
import os


def explore_csv(filepath):
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total rows: {len(rows)}")
    print("Columns:", reader.fieldnames)

    for i, row in enumerate(rows[:3]):
        print(f"\nRow {i}:")
        print(f"  patient_id: {row['patient_id']}")
        print(f"  pathology: {row['pathology']}")
        print(f"  image file path: {row['image file path']}")

        # Parse DICOM path
        parts = row["image file path"].split("/")
        if len(parts) >= 3:
            series_uid = parts[1]
            print(f"  series_uid: {series_uid}")

            # Check if folder exists
            jpeg_dir = os.path.join("data", "jpeg", series_uid)
            if os.path.exists(jpeg_dir):
                print(f"  JPEG dir exists: {jpeg_dir}")
                jpegs = os.listdir(jpeg_dir)
                print(f"  JPEGs: {jpegs}")
            else:
                print(f"  JPEG dir NOT found: {jpeg_dir}")

        print(f"  cropped image file path: {row['cropped image file path']}")
        print(f"  ROI mask file path: {row['ROI mask file path']}")


if __name__ == "__main__":
    csv_file = "data/csv/mass_case_description_train_set.csv"
    if os.path.exists(csv_file):
        explore_csv(csv_file)
    else:
        print(f"File not found: {csv_file}")
