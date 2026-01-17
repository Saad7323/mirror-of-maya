import os
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_DIR = os.path.join(BASE_DIR, "data", "raw", "original")
MODIFIED_DIR = os.path.join(BASE_DIR, "data", "raw", "modified")

print("ORIGINAL_DIR:", ORIGINAL_DIR)
print("MODIFIED_DIR:", MODIFIED_DIR)

os.makedirs(MODIFIED_DIR, exist_ok=True)

files = os.listdir(ORIGINAL_DIR)
print("Files found:", files)

if len(files) == 0:
    print("❌ No images found in original folder")
    exit(1)

for img_name in files:
    img_path = os.path.join(ORIGINAL_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Could not read image: {img_name}")
        continue

    base = os.path.splitext(img_name)[0]

    resized = cv2.resize(
        img,
        (int(img.shape[1] * 0.8), int(img.shape[0] * 0.8))
    )
    cv2.imwrite(os.path.join(MODIFIED_DIR, f"{base}_resize.jpg"), resized)

    cv2.imwrite(
        os.path.join(MODIFIED_DIR, f"{base}_jpeg.jpg"),
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), 40]
    )

    h, w, _ = img.shape
    cropped = img[int(0.1*h):int(0.9*h), int(0.1*w):int(0.9*w)]
    cv2.imwrite(os.path.join(MODIFIED_DIR, f"{base}_crop.jpg"), cropped)

print("✅ Near-duplicate images created successfully.")
