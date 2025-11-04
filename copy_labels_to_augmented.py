# copy_labels_to_augmented.py
import os
import shutil
from glob import glob
from tqdm import tqdm

BASE = "bdd100k"
YOLO_LABELS = os.path.join(BASE, "yolo_format")
AUG_IMAGES = os.path.join(BASE, "augmented")
AUG_LABELS = os.path.join(BASE, "augmented_labels")

# Weather augmentation suffixes
AUG_TYPES = ['fog', 'rain', 'snow', 'lowlight']

def copy_labels_for_split(split):
    print(f"\n{'='*60}")
    print(f"Processing {split.upper()} split")
    print('='*60)
    
    label_dir = os.path.join(YOLO_LABELS, split)
    aug_label_dir = os.path.join(AUG_LABELS, split)
    
    os.makedirs(aug_label_dir, exist_ok=True)
    
    # Get all original labels
    label_files = glob(os.path.join(label_dir, "*.txt"))
    
    if not label_files:
        print(f"⚠️  No labels found in {label_dir}")
        return 0
    
    print(f"📂 Found {len(label_files):,} original labels")
    
    copied = 0
    
    for label_path in tqdm(label_files, desc=f"Copying {split}"):
        try:
            # Get base name (without extension)
            base_name = os.path.splitext(os.path.basename(label_path))[0]
            
            # Copy for each augmentation type
            for aug_type in AUG_TYPES:
                # New augmented label name
                aug_label_name = f"{base_name}_{aug_type}.txt"
                aug_label_path = os.path.join(aug_label_dir, aug_label_name)
                
                # Copy the original label (same bounding boxes apply to augmented images)
                shutil.copy2(label_path, aug_label_path)
                copied += 1
        
        except Exception as e:
            print(f"Error copying {label_path}: {e}")
    
    print(f"✅ Copied {copied:,} augmented labels")
    return copied

def main():
    print("\n" + "="*60)
    print("📋 Copying Labels to Augmented Images")
    print("="*60)
    print("\nThis creates labels for augmented images:")
    print("  - Original: image.jpg → image.txt")
    print("  - Fog: image_fog.jpg → image_fog.txt")
    print("  - Rain: image_rain.jpg → image_rain.txt")
    print("  - Snow: image_snow.jpg → image_snow.txt")
    print("  - Lowlight: image_lowlight.jpg → image_lowlight.txt")
    
    total_copied = 0
    
    for split in ['train', 'val', 'test']:
        copied = copy_labels_for_split(split)
        total_copied += copied
    
    print("\n" + "="*60)
    print("✅ LABEL COPYING COMPLETE!")
    print("="*60)
    print(f"\nTotal labels copied: {total_copied:,}")
    
    # Verify
    print("\n🔍 Verification:")
    for split in ['train', 'val', 'test']:
        aug_label_dir = os.path.join(AUG_LABELS, split)
        if os.path.exists(aug_label_dir):
            count = len(glob(os.path.join(aug_label_dir, "*.txt")))
            print(f"   {split}: {count:,} augmented labels")
    
    print("\n✅ Next step: Create dataset YAML file")

if __name__ == "__main__":
    main()
