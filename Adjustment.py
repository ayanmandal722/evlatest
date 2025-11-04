import os
import json
from glob import glob
from PIL import Image
from tqdm import tqdm

# Classes in BDD100K (detection)
BDD_CLASSES = [
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "traffic light", "traffic sign"
]

base_dir = "bdd100k"

# Try to find the image and label folders automatically
def find_subfolder(base, keyword):
    for root, dirs, files in os.walk(base):
        if keyword in root.lower():
            return root
    return None

images_dir = find_subfolder(base_dir, "image")
labels_dir = find_subfolder(base_dir, "label")

if not images_dir or not labels_dir:
    raise FileNotFoundError("❌ Could not auto-detect image or label folder. Check dataset structure.")

output_dir = os.path.join(base_dir, "yolo_format")
os.makedirs(output_dir, exist_ok=True)

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return (x_center * dw, y_center * dh, w * dw, h * dh)

def process_split(split):
    print(f"\n📂 Processing {split} set...")

    split_img_dir = os.path.join(images_dir, split)
    split_label_dir = os.path.join(labels_dir, split)
    split_out_dir = os.path.join(output_dir, split)
    os.makedirs(split_out_dir, exist_ok=True)

    json_files = glob(os.path.join(split_label_dir, "*.json"))

    if not json_files:
        print(f"⚠️ No JSON files found in {split_label_dir}")
        return

    for json_file in tqdm(json_files, desc=f"{split} conversion", unit="file"):
        with open(json_file, "r") as f:
            data = json.load(f)

        img_name = data["name"]
        img_path = os.path.join(split_img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        label_out = os.path.join(split_out_dir, img_name.replace(".jpg", ".txt"))
        with open(label_out, "w") as f_out:
            for obj in data.get("labels", []):
                category = obj["category"]
                if category not in BDD_CLASSES:
                    continue
                cls_id = BDD_CLASSES.index(category)

                if "box2d" in obj:
                    box = obj["box2d"]
                    x_min, y_min, x_max, y_max = box["x1"], box["y1"], box["x2"], box["y2"]
                    bb = convert_bbox((w, h), (x_min, y_min, x_max, y_max))
                    f_out.write(f"{cls_id} " + " ".join([str(a) for a in bb]) + "\n")

for split in ["train", "val", "test"]:
    process_split(split)

print("\n✅ Conversion to YOLO format completed!")
