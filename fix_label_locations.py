# fix_label_locations.py
import os
import shutil
from glob import glob
from tqdm import tqdm

BASE = "bdd100k"
AUG_IMAGES = os.path.join(BASE, "augmented")
AUG_LABELS = os.path.join(BASE, "augmented_labels")

def move_labels_to_images(split):
    print(f"\n{'='*60}")
    print(f"Moving {split} labels to image folder")
    print('='*60)
    
    label_src = os.path.join(AUG_LABELS, split)
    img_dst = os.path.join(AUG_IMAGES, split)
    
    if not os.path.exists(label_src):
        print(f"❌ Label folder not found: {label_src}")
        return 0
    
    label_files = glob(os.path.join(label_src, "*.txt"))
    
    if not label_files:
        print(f"❌ No labels found in {label_src}")
        return 0
    
    print(f"📂 Found {len(label_files):,} labels")
    print(f"📋 Moving to {img_dst}")
    
    moved = 0
    for label_path in tqdm(label_files, desc=f"Moving {split}"):
        try:
            label_name = os.path.basename(label_path)
            dst_path = os.path.join(img_dst, label_name)
            shutil.copy2(label_path, dst_path)
            moved += 1
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"✅ Moved {moved:,} labels")
    return moved

print("="*60)
print("🔧 Moving Labels to Image Folders")
print("="*60)
print("\nYOLO expects labels in SAME folder as images!")

total = 0
for split in ['train', 'val', 'test']:
    total += move_labels_to_images(split)

print(f"\n{'='*60}")
print(f"✅ Total labels moved: {total:,}")
print("="*60)

# Verify
print("\n🔍 Verification:")
for split in ['train', 'val', 'test']:
    img_dir = os.path.join("bdd100k", "augmented", split)
    imgs = len(glob(os.path.join(img_dir, "*.jpg")))
    lbls = len(glob(os.path.join(img_dir, "*.txt")))
    print(f"   {split}: {imgs:,} images, {lbls:,} labels")
