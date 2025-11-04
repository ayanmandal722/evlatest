# verify_dataset.py
import os
from glob import glob

BASE = "bdd100k"
IMAGES_ROOT = os.path.join(BASE, "images", "100k")
AUG_IMAGES_ROOT = os.path.join(BASE, "augmented")
YOLO_LABELS_ROOT = os.path.join(BASE, "yolo_format")
AUG_LABELS_ROOT = os.path.join(BASE, "augmented_labels")

SPLITS = ['train', 'val', 'test']

def count_files(directory, extensions):
    """Count files with given extensions in directory"""
    count = 0
    if os.path.exists(directory):
        for ext in extensions:
            count += len(glob(os.path.join(directory, f"*{ext}")))
    return count

print("📊 DATASET VERIFICATION REPORT")
print("=" * 50)

for split in SPLITS:
    print(f"\n🔍 {split.upper()} SPLIT:")
    print("-" * 30)
    
    # Original images
    img_dir = os.path.join(IMAGES_ROOT, split)
    orig_images = count_files(img_dir, ['.jpg', '.jpeg', '.png'])
    
    # Augmented images
    aug_img_dir = os.path.join(AUG_IMAGES_ROOT, split)
    aug_images = count_files(aug_img_dir, ['.jpg', '.jpeg', '.png'])
    
    # Original YOLO labels
    orig_lbl_dir = os.path.join(YOLO_LABELS_ROOT, split)
    orig_labels = count_files(orig_lbl_dir, ['.txt'])
    
    # Augmented YOLO labels
    aug_lbl_dir = os.path.join(AUG_LABELS_ROOT, split)
    aug_labels = count_files(aug_lbl_dir, ['.txt'])
    
    print(f"  Original images: {orig_images:,}")
    print(f"  Augmented images: {aug_images:,} (should be {orig_images * 4:,})")
    print(f"  Original labels: {orig_labels:,}")
    print(f"  Augmented labels: {aug_labels:,} (should be {orig_labels * 4:,})")
    
    if aug_images > 0:
        print(f"  ✅ Augmentation success: {aug_images/orig_images:.1f}x images")
    else:
        print(f"  ❌ No augmented images found!")
    
    if orig_labels == 0:
        print(f"  ❌ No original YOLO labels - need to fix conversion!")

print("\n" + "=" * 50)
print("📋 SUMMARY:")
print(f"Total original images: {count_files(os.path.join(IMAGES_ROOT, '*'), ['*.jpg', '*.jpeg', '*.png']):,}")
print(f"Total augmented images: {count_files(os.path.join(AUG_IMAGES_ROOT, '*'), ['*.jpg', '*.jpeg', '*.png']):,}")
print(f"Total YOLO labels: {count_files(os.path.join(YOLO_LABELS_ROOT, '*'), ['*.txt']):,}")

if count_files(os.path.join(YOLO_LABELS_ROOT, '*'), ['*.txt']) == 0:
    print("\n🚨 CRITICAL: No YOLO labels found! Need to fix conversion script.")
    print("Next step: Run fixed conversion script")
else:
    print("\n✅ All files ready for training!")
    print("Next step: Create dataset YAML and start training")
