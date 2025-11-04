import os
from glob import glob
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Paths for both datasets
BASE = "bdd10k"
BASE1 = "bdd100k"
DS_10K = {
    "name": "BDD10K 10K",
    "images": os.path.join(BASE, "images_10k"),
    "labels": os.path.join(BASE, "yolo_format")
}
DS_100K = {
    "name": "BDD100K 100K",
    "images": os.path.join(BASE1, "100k"),
    "labels": os.path.join(BASE1, "yolo_format_100k")  # <-- you’ll generate this from the 100k labels
}

REPORT_DIR = "report"
REPORT_IMG_DIR = os.path.join(REPORT_DIR, "compare_10k_100k")
os.makedirs(REPORT_IMG_DIR, exist_ok=True)

SPLITS = ["train", "val", "test"]
CLASSES = [
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "traffic light", "traffic sign"
]

def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return len(glob(os.path.join(folder, "*.jpg"))) + len(glob(os.path.join(folder, "*.png")))

def parse_yolo_labels(folder):
    counter = Counter()
    if not os.path.isdir(folder):
        return counter
    files = glob(os.path.join(folder, "*.txt"))
    for f in tqdm(files, desc=f"Parsing {folder}"):
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

def dataset_summary(ds):
    summary = {}
    for split in SPLITS:
        img_dir = os.path.join(ds["images"], split)
        lbl_dir = os.path.join(ds["labels"], split)
        orig_count = count_images(img_dir)
        lbls = parse_yolo_labels(lbl_dir)
        summary[split] = {
            "images": orig_count,
            "labels": sum(lbls.values()),
            "dist": lbls
        }
    return summary

def plot_compare_bar(dist1, dist2, name1, name2, split, out_path):
    labels = CLASSES
    values1 = [dist1.get(l, 0) for l in labels]
    values2 = [dist2.get(l, 0) for l in labels]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12,5))
    plt.bar(x - width/2, values1, width, label=name1)
    plt.bar(x + width/2, values2, width, label=name2)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel("Count")
    plt.title(f"{split.capitalize()} split - Class distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---- MAIN ----
print("Generating comparison...")
sum10k = dataset_summary(DS_10K)
sum100k = dataset_summary(DS_100K)

md_path = os.path.join(REPORT_DIR, "compare_10k_vs_100k.md")
with open(md_path, "w", encoding="utf-8") as md:
    md.write("# BDD100K 10K vs 100K Comparison\n\n")
    md.write("| Split | Dataset | #Images | #Labels |\n")
    md.write("|-------|----------|---------|---------|\n")
    for split in SPLITS:
        md.write(f"| {split} | {DS_10K['name']} | {sum10k[split]['images']} | {sum10k[split]['labels']} |\n")
        md.write(f"| {split} | {DS_100K['name']} | {sum100k[split]['images']} | {sum100k[split]['labels']} |\n")

    md.write("\n## Class distribution comparisons\n\n")
    for split in SPLITS:
        chart_path = os.path.join(REPORT_IMG_DIR, f"{split}_compare.png")
        plot_compare_bar(sum10k[split]["dist"], sum100k[split]["dist"], DS_10K["name"], DS_100K["name"], split, chart_path)
        rel = os.path.relpath(chart_path, REPORT_DIR)
        md.write(f"### {split.capitalize()} split\n\n")
        md.write(f"![{split} compare]({rel})\n\n")

print(f"✅ Comparison markdown saved: {md_path}")
print(f"Charts in: {REPORT_IMG_DIR}")
