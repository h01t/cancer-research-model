import csv
import os

# Load dicom_info.csv mapping
dicom_info_path = "data/csv/dicom_info.csv"
if not os.path.exists(dicom_info_path):
    print("dicom_info.csv not found")
    exit()

# Build mapping from SOPInstanceUID to JPEG path
sop_to_jpeg = {}
with open(dicom_info_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sop = row["SOPInstanceUID"]
        jpeg_path = row["image_path"]
        # Remove CBIS-DDSM/ prefix if present
        if jpeg_path.startswith("CBIS-DDSM/"):
            jpeg_path = jpeg_path[10:]  # remove prefix
        sop_to_jpeg[sop] = jpeg_path

print(f"Loaded {len(sop_to_jpeg)} SOP to JPEG mappings")

# Test with mass training CSV
mass_csv = "data/csv/mass_case_description_train_set.csv"
with open(mass_csv, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"\nTesting {len(rows)} mass rows")
found = 0
not_found = 0

for i, row in enumerate(rows[:10]):
    dicom_path = row["image file path"]
    # Parse SOPInstanceUID: last component before .dcm
    # Format: .../SOPInstanceUID/000000.dcm
    parts = dicom_path.split("/")
    if len(parts) >= 2:
        sop = parts[-2]  # second last component is SOPInstanceUID?
        # Actually pattern: SeriesInstanceUID/SOPInstanceUID/000000.dcm
        # So SOPInstanceUID is second last
        sop = parts[-2]
        if sop in sop_to_jpeg:
            found += 1
            jpeg = sop_to_jpeg[sop]
            # Check if file exists
            full_path = os.path.join("data", jpeg)
            exists = os.path.exists(full_path)
            print(f"Row {i}: SOP {sop} -> JPEG {jpeg} exists: {exists}")
        else:
            not_found += 1
            print(f"Row {i}: SOP {sop} NOT FOUND in dicom_info")

print(f"\nFound: {found}, Not found: {not_found}")

# Also check SeriesDescription
print("\nChecking SeriesDescription for found rows...")
for i, row in enumerate(rows[:3]):
    dicom_path = row["image file path"]
    parts = dicom_path.split("/")
    if len(parts) >= 2:
        sop = parts[-2]
        if sop in sop_to_jpeg:
            # Look up row in dicom_info
            with open(dicom_info_path, "r") as f:
                reader2 = csv.DictReader(f)
                for info_row in reader2:
                    if info_row["SOPInstanceUID"] == sop:
                        print(
                            f"Row {i}: SeriesDescription: {info_row['SeriesDescription']}"
                        )
                        break
