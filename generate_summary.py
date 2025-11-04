import os
from glob import glob
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from shutil import copy2
from PIL import Image
from tqdm import tqdm   # ✅ NEW

# ---------- CONFIG ----------
BASE = "bdd100k"
ORIG_IMAGES = os.path.join(BASE, "100k")
YOLO_LABELS = os.path.join(BASE, "yolo_format")
AUG_IMAGES = os.path.join(BASE, "augmented")
AUG_LABELS = os.path.join(BASE, "augmented_labels")
REPORT_DIR = "report"
REPORT_IMG_DIR = os.path.join(REPORT_DIR, "images")
SPLITS = ["train", "val", "test"]
CLASSES = [
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "traffic light", "traffic sign"
]
SAMPLE_PER_SPLIT = 3
# -----------------------------

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(REPORT_IMG_DIR, exist_ok=True)

def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return len(glob(os.path.join(folder, "*.jpg"))) + len(glob(os.path.join(folder, "*.png")))

def parse_yolo_labels(folder):
    counter = Counter()
    if not os.path.isdir(folder):
        return counter
    files = glob(os.path.join(folder, "*.txt"))
    for f in tqdm(files, desc=f"Parsing labels in {os.path.basename(folder)}"):
        with open(f, "r", encoding="utf-8") as fo:
            for line in fo:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls_id = int(parts[0])
                except:
                    continue
                if 0 <= cls_id < len(CLASSES):
                    counter[CLASSES[cls_id]] += 1
    return counter

def plot_and_save_distribution(counter, title, out_path):
    labels = CLASSES
    counts = [counter.get(l, 0) for l in labels]
    y_pos = np.arange(len(labels))

    plt.figure(figsize=(10,4))
    plt.bar(y_pos, counts)
    plt.xticks(y_pos, labels, rotation=45, ha='right')
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def pick_samples_for_split(split, n_samples=3):
    src_dir = os.path.join(ORIG_IMAGES, split)
    if not os.path.isdir(src_dir):
        return []
    imgs = sorted(glob(os.path.join(src_dir, "*.jpg")) + glob(os.path.join(src_dir, "*.png")))
    return imgs[:n_samples]

def find_aug_variants_for_image(img_path, split):
    base = os.path.splitext(os.path.basename(img_path))[0]
    aug_dir = os.path.join(AUG_IMAGES, split)
    candidates = []
    if not os.path.isdir(aug_dir):
        return candidates
    patterns = [f"{base}_fog.jpg", f"{base}_rain.jpg", f"{base}_lowlight.jpg", f"{base}_snow.jpg"]
    for p in patterns:
        pfull = os.path.join(aug_dir, p)
        if os.path.exists(pfull):
            candidates.append(pfull)
    return candidates

# Build summary data and plots
summary = {}

for split in SPLITS:
    print(f"\n=== Processing split: {split} ===")
    orig_img_dir = os.path.join(ORIG_IMAGES, split)
    yolo_dir = os.path.join(YOLO_LABELS, split)
    aug_img_dir = os.path.join(AUG_IMAGES, split)
    aug_lbl_dir = os.path.join(AUG_LABELS, split)

    orig_count = count_images(orig_img_dir)
    aug_count = count_images(aug_img_dir)
    orig_lbls = parse_yolo_labels(yolo_dir)
    aug_lbls = parse_yolo_labels(aug_lbl_dir)

    summary[split] = {
        "orig_images": orig_count,
        "aug_images": aug_count,
        "orig_labels_total": sum(orig_lbls.values()),
        "aug_labels_total": sum(aug_lbls.values()),
        "orig_label_dist": orig_lbls,
        "aug_label_dist": aug_lbls
    }

    plot_and_save_distribution(orig_lbls, f"{split} - Original Class Distribution", os.path.join(REPORT_IMG_DIR, f"{split}_orig_dist.png"))
    plot_and_save_distribution(aug_lbls, f"{split} - Augmented Class Distribution", os.path.join(REPORT_IMG_DIR, f"{split}_aug_dist.png"))

# Pick sample images and copy to report/images
sample_table_rows = []
for split in SPLITS:
    samples = pick_samples_for_split(split, SAMPLE_PER_SPLIT)
    for img_path in tqdm(samples, desc=f"Copying samples for {split}"):
        base = os.path.splitext(os.path.basename(img_path))[0]
        dest_orig = os.path.join(REPORT_IMG_DIR, f"{split}_{base}.jpg")
        try:
            copy2(img_path, dest_orig)
            orig_rel = os.path.relpath(dest_orig, REPORT_DIR)
        except Exception:
            orig_rel = ""
        variants = find_aug_variants_for_image(img_path, split)
        fog_rel = rain_rel = low_rel = snow_rel = ""
        for v in variants:
            name = os.path.basename(v)
            if name.endswith("_fog.jpg"):
                dst = os.path.join(REPORT_IMG_DIR, f"{split}_{base}_fog.jpg"); copy2(v, dst); fog_rel = os.path.relpath(dst, REPORT_DIR)
            elif name.endswith("_rain.jpg"):
                dst = os.path.join(REPORT_IMG_DIR, f"{split}_{base}_rain.jpg"); copy2(v, dst); rain_rel = os.path.relpath(dst, REPORT_DIR)
            elif name.endswith("_lowlight.jpg"):
                dst = os.path.join(REPORT_IMG_DIR, f"{split}_{base}_lowlight.jpg"); copy2(v, dst); low_rel = os.path.relpath(dst, REPORT_DIR)
            elif name.endswith("_snow.jpg"):
                dst = os.path.join(REPORT_IMG_DIR, f"{split}_{base}_snow.jpg"); copy2(v, dst); snow_rel = os.path.relpath(dst, REPORT_DIR)
        sample_table_rows.append((split, orig_rel, fog_rel, rain_rel, low_rel, snow_rel))

# Write markdown
md_path = os.path.join(REPORT_DIR, "comparative_summary.md")
with open(md_path, "w", encoding="utf-8") as md:
    md.write("# Comparative Dataset Report\n\n")
    md.write("This report compares the original BDD100K samples with the synthetic augmented dataset (fog, rain, low-light, snow).\n\n")
    md.write("## 1. Dataset Overview\n\n")
    md.write("| Split | Original images | Augmented images | Original labels | Augmented labels |\n")
    md.write("|-------|-----------------|------------------|-----------------|------------------|\n")
    for split in SPLITS:
        s = summary[split]
        md.write(f"| {split} | {s['orig_images']} | {s['aug_images']} | {s['orig_labels_total']} | {s['aug_labels_total']} |\n")
    md.write("\n")

    md.write("## 2. Class distributions (bar charts)\n\n")
    for split in SPLITS:
        orig_chart = f"images/{split}_orig_dist.png"
        aug_chart = f"images/{split}_aug_dist.png"
        if os.path.exists(os.path.join(REPORT_DIR, orig_chart)):
            md.write(f"### {split.capitalize()}\n\n")
            md.write(f"**Original class distribution**  \n\n")
            md.write(f"![{split} original]({orig_chart})\n\n")
            md.write(f"**Augmented class distribution**  \n\n")
            md.write(f"![{split} augmented]({aug_chart})\n\n")

    md.write("## 3. Sample images (original vs augmentations)\n\n")
    md.write("Each row shows an original sample and its augmented variants (fog, rain, lowlight, snow) if available.\n\n")
    md.write("| Split | Original | Fog | Rain | Low-light | Snow |\n")
    md.write("|-------|----------|-----|------|-----------|------|\n")
    for row in sample_table_rows:
        split, orig_rel, fog_rel, rain_rel, low_rel, snow_rel = row
        def img_md(path):
            if path and os.path.exists(os.path.join(REPORT_DIR, path)):
                return f"![img]({path})"
            else:
                return "N/A"
        md.write(f"| {split} | {img_md(orig_rel)} | {img_md(fog_rel)} | {img_md(rain_rel)} | {img_md(low_rel)} | {img_md(snow_rel)} |\n")

    md.write("\n## 4. Observations & Notes\n\n")
    md.write("- Augmentation expands dataset diversity and preserves label distribution shape in most classes.\n")
    md.write("- Please check for any missing labels (some augmented images may not have copied labels if original label files were missing).\n")
    md.write("- Next steps: train baseline detection model (YOLO) on the augmented+original data and benchmark on real adverse-weather datasets.\n")

print("Report generated ->", md_path)
print("Images and charts saved in ->", REPORT_IMG_DIR)
