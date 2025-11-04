import os
import json
from glob import glob
from PIL import Image
from tqdm import tqdm
import shutil
import cv2
import random
import numpy as np

# --------- CONFIG ----------
BASE = "bdd100k"
IMAGES_ROOT = os.path.join(BASE, "images_10k")   # must contain train/ val/ test
LABELS_ROOT = os.path.join(BASE, "labels")       # 100k per-image JSONs (train/val/test)
BACKUP_LABELS = os.path.join(BASE, "labels_kept_backup")  # moved-away files go here
YOLO_OUT = os.path.join(BASE, "yolo_format")     # will be created/filled
AUG_IMAGES_OUT = os.path.join(BASE, "augmented")
AUG_LABELS_OUT = os.path.join(BASE, "augmented_labels")
SPLITS = ["train", "val", "test"]
TEST_LIMIT = None   # set int for quick testing per split

BDD_CLASSES = [
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "traffic light", "traffic sign"
]

# --------- Utils ----------
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def list_images_for_split(split):
    d = os.path.join(IMAGES_ROOT, split)
    files = []
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG"):
        files += glob(os.path.join(d, ext))
    return sorted(files)

def try_image_variants(base_name, split_img_dir):
    variants = [base_name, base_name + ".jpg", base_name + ".png", base_name + ".jpeg"]
    for v in variants:
        p = os.path.join(split_img_dir, v)
        if os.path.exists(p):
            return os.path.basename(p)
    for v in variants:
        p = os.path.join(split_img_dir, v.lower())
        if os.path.exists(p):
            return os.path.basename(p.lower())
    return None

def convert_bbox_to_yolo(w,h,box):
    x1,y1,x2,y2 = box
    if max(x1,y1,x2,y2) <= 1.01:
        x1 *= w; x2 *= w; y1 *= h; y2 *= h
    x1 = max(0.0, min(x1, w)); x2 = max(0.0, min(x2, w))
    y1 = max(0.0, min(y1, h)); y2 = max(0.0, min(y2, h))
    bw = x2 - x1; bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    return cx / w, cy / h, bw / w, bh / h

# ------------- Step 1: Filter label JSONs to images -------------
def filter_labels():
    ensure_dir(BACKUP_LABELS)
    moved = 0
    for split in SPLITS:
        img_dir = os.path.join(IMAGES_ROOT, split)
        lbl_dir = os.path.join(LABELS_ROOT, split)
        backup_split = os.path.join(BACKUP_LABELS, split)
        ensure_dir(backup_split)
        if not os.path.isdir(lbl_dir):
            print(f"Labels split folder not found: {lbl_dir} (skipping)")
            continue
        # images basenames present
        img_files = list_images_for_split(split)
        img_basenames = {os.path.splitext(os.path.basename(p))[0] for p in img_files}
        print(f"[Filter] Split={split}: images found={len(img_basenames)}, label jsons={len(glob(os.path.join(lbl_dir, '*.json')))}")
        for jf in glob(os.path.join(lbl_dir, "*.json")):
            base = os.path.splitext(os.path.basename(jf))[0]
            if base not in img_basenames:
                # move it to backup
                dst = os.path.join(backup_split, os.path.basename(jf))
                shutil.move(jf, dst)
                moved += 1
    print(f"[Filter] Moved {moved} label JSON files to {BACKUP_LABELS}")

# ------------- Step 2: Convert per-image JSONs -> YOLO -------------
def convert_jsons_to_yolo():
    ensure_dir(YOLO_OUT)
    total_images = 0
    total_boxes = 0
    for split in SPLITS:
        img_dir = os.path.join(IMAGES_ROOT, split)
        lbl_dir = os.path.join(LABELS_ROOT, split)
        out_dir = os.path.join(YOLO_OUT, split)
        ensure_dir(out_dir)
        json_files = glob(os.path.join(lbl_dir, "*.json"))
        if TEST_LIMIT:
            json_files = json_files[:TEST_LIMIT]
        if not json_files:
            print(f"[Convert] No JSONs in {lbl_dir}, skipping.")
            continue
        imgs_with_ann = 0
        boxes_written = 0
        for jf in tqdm(json_files, desc=f"Convert {split}", unit="file"):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            name_base = data.get("name", "")
            if not name_base:
                continue
            img_name = try_image_variants(name_base, img_dir)
            if img_name is None:
                continue
            img_path = os.path.join(img_dir, img_name)
            try:
                with Image.open(img_path) as im:
                    w,h = im.size
            except Exception:
                continue
            # collect objects
            objs =
