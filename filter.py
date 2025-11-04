# filter_labels_to_images.py
# Usage: python filter_labels_to_images.py
# This script moves label JSON files that DO NOT match an existing image
# into bdd100k/labels_kept_backup/<split>/ so only matching labels remain
# in bdd100k/labels/<split>.

import os
from glob import glob
import shutil

BASE = "bdd100k"
IMAGES_ROOT = os.path.join(BASE, "images_10k")   # where your 10k images live
LABELS_ROOT = os.path.join(BASE, "labels")      # currently contains ~100k JSONs (train/val/test)
BACKUP_ROOT = os.path.join(BASE, "labels_kept_backup")
SPLITS = ["train", "val", "test"]

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def gather_image_basenames(img_dir):
    basenames = set()
    if not os.path.isdir(img_dir):
        return basenames
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG"):
        for p in glob(os.path.join(img_dir, ext)):
            basenames.add(os.path.splitext(os.path.basename(p))[0])
    return basenames

def filter_split(split):
    img_dir = os.path.join(IMAGES_ROOT, split)
    lbl_dir = os.path.join(LABELS_ROOT, split)
    backup_dir = os.path.join(BACKUP_ROOT, split)
    ensure_dir(backup_dir)

    if not os.path.isdir(lbl_dir):
        print(f"[WARN] label folder not found for split '{split}': {lbl_dir}  (skipping)")
        return 0, 0

    img_basenames = gather_image_basenames(img_dir)
    print(f"[INFO] split='{split}': {len(img_basenames)} image basenames found in {img_dir}")

    json_files = glob(os.path.join(lbl_dir, "*.json"))
    moved = 0
    kept = 0
    for jf in json_files:
        base = os.path.splitext(os.path.basename(jf))[0]
        if base in img_basenames:
            # keep it in place
            kept += 1
        else:
            # move to backup folder
            dst = os.path.join(backup_dir, os.path.basename(jf))
            shutil.move(jf, dst)
            moved += 1
    print(f"[DONE] split='{split}': kept={kept}, moved_to_backup={moved}")
    return kept, moved

def main():
    print("=== Starting filtering labels to match existing images ===")
    ensure_dir(BACKUP_ROOT)
    total_kept = total_moved = 0
    for sp in SPLITS:
        kept, moved = filter_split(sp)
        total_kept += kept
        total_moved += moved
    print("=== Summary ===")
    print(f"Total labels kept (matching images): {total_kept}")
    print(f"Total label files moved to backup: {total_moved}")
    print(f"Backup folder: {os.path.abspath(BACKUP_ROOT)}")
    print("Now your bdd100k/labels/<split>/ folders should contain only labels for images you have.")
    print("Next steps (recommended):")
    print("  1) Run your YOLO conversion script (create .txt from remaining JSONs).")
    print("  2) Run augment_dataset.py again to copy labels to augmented_labels.")
    print("  3) Re-run diagnosis/report scripts to confirm counts.")
    print("If you prefer to restore any moved file, look inside the backup folder above.")

if __name__ == '__main__':
    main()
