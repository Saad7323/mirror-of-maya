import os
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORIGINAL_DIR = os.path.join(BASE_DIR, "data", "raw", "original")
MODIFIED_DIR = os.path.join(BASE_DIR, "data", "raw", "modified")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "pairs.csv")

pairs = []

original_images = os.listdir(ORIGINAL_DIR)
modified_images = os.listdir(MODIFIED_DIR)

# -------- Positive pairs (near-duplicates) --------
for orig in original_images:
    base = os.path.splitext(orig)[0]

    for mod in modified_images:
        if base in mod:
            pairs.append([
                f"original/{orig}",
                f"modified/{mod}",
                1
            ])

# -------- Negative pairs (non-duplicates) --------
for i in range(len(original_images) - 1):
    pairs.append([
        f"original/{original_images[i]}",
        f"original/{original_images[i + 1]}",
        0
    ])

# Write CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_1", "image_2", "label"])
    writer.writerows(pairs)

print(f"pairs.csv created with {len(pairs)} pairs")
