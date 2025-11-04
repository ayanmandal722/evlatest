# diagnose_labels.py
import os
from glob import glob
import csv

BASE = "bdd100k"
IMAGES_ROOT = os.path.join(BASE, "images_10k")
YOLO_ROOT = os.path.join(BASE, "yolo_format")
AUG_IMG_ROOT = os.path.join(BASE, "augmented")
AUG_LBL_ROOT = os.path.join(BASE, "augmented_labels")
SPLITS = ["train", "val", "test"]

def list_images(split):
    d = os.path.join(IMAGES_ROOT, split)
    if not os.path.isdir(d):
        return []
    files = []
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG"):
        files.extend(glob(os.path.join(d, ext)))
    return sorted(files)

def list_labels(split):
    d = os.path.join(YOLO_ROOT, split)
    if not os.path.isdir(d):
        return []
    return sorted(glob(os.path.join(d, "*.txt")))

def basename(p): return os.path.splitext(os.path.basename(p))[0]

missing_report = []

total_imgs = 0
total_lbls = 0
for sp in SPLITS:
    imgs = list_images(sp)
    lbls = list_labels(sp)
    img_basenames = {basename(p) for p in imgs}
    lbl_basenames = {basename(p) for p in lbls}
    total_imgs += len(imgs)
    total_lbls += len(lbls)
    missing = sorted(list(img_basenames - lbl_basenames))
    print(f"\n=== {sp.upper()} ===")
    print(f"Images: {len(imgs)}, Labels: {len(lbls)}, Missing labels for images: {len(missing)}")
    if len(imgs)>0:
        print("Sample images:", imgs[:3])
    if len(lbls)>0:
        print("Sample labels:", lbls[:3])
    if missing:
        print("Examples of images missing labels:", missing[:10])
    for name in missing:
        missing_report.append((sp, name))

print(f"\nTOTAL images: {total_imgs}, TOTAL label files: {total_lbls}")

# Save missing report
if missing_report:
    out_csv = "missing_labels_report.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split","image_basename_missing_label"])
        w.writerows(missing_report)
    print(f"\nMissing label report saved -> {out_csv}")
else:
    print("\nNo missing labels detected (labels exist for all images).")
