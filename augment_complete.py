# augment_complete.py
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import shutil

BASE = "bdd100k"
IMAGES_ROOT = os.path.join(BASE, "images", "100k")
LABELS_ROOT = os.path.join(BASE, "yolo_format")
AUG_IMAGES_ROOT = os.path.join(BASE, "augmented")
AUG_LABELS_ROOT = os.path.join(BASE, "augmented_labels")

# Augmentation functions
def add_fog(img, intensity=0.7):
    """Add fog effect"""
    h, w = img.shape[:2]
    fog_layer = np.ones((h, w, 3), dtype=np.uint8) * 200
    alpha = intensity
    fogged = cv2.addWeighted(img, 1-alpha, fog_layer, alpha, 0)
    return fogged

def add_rain(img, intensity=0.8):
    """Add rain effect"""
    rain_img = img.copy()
    h, w = img.shape[:2]
    
    # Create rain drops
    num_drops = int(1200 * intensity)
    for _ in range(num_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(15, 35)
        thickness = 1
        cv2.line(rain_img, (x, y), (x+2, y+length), (200, 200, 200), thickness)
    
    # Blend
    rain_img = cv2.addWeighted(img, 0.75, rain_img, 0.25, 0)
    
    # Add slight blur for realism
    rain_img = cv2.GaussianBlur(rain_img, (3, 3), 0)
    return rain_img

def add_snow(img, intensity=0.6):
    """Add snow effect"""
    snow_img = img.copy()
    h, w = img.shape[:2]
    
    # Create snowflakes
    num_flakes = int(1000 * intensity)
    for _ in range(num_flakes):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        size = np.random.randint(2, 5)
        cv2.circle(snow_img, (x, y), size, (255, 255, 255), -1)
    
    # Blend and add brightness
    snow_img = cv2.addWeighted(img, 0.7, snow_img, 0.3, 10)
    return snow_img

def add_lowlight(img, factor=0.35):
    """Simulate low-light/night conditions"""
    dark_img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
    
    # Add slight blue tint for night effect
    blue_tint = np.zeros_like(dark_img)
    blue_tint[:, :, 0] = 20  # Blue channel
    dark_img = cv2.addWeighted(dark_img, 0.9, blue_tint, 0.1, 0)
    
    return dark_img

def augment_image(img_path, aug_type):
    """Apply augmentation based on type"""
    img = cv2.imread(img_path)
    
    if img is None:
        return None
    
    if aug_type == 'fog':
        return add_fog(img)
    elif aug_type == 'rain':
        return add_rain(img)
    elif aug_type == 'snow':
        return add_snow(img)
    elif aug_type == 'lowlight':
        return add_lowlight(img)
    else:
        return img

def process_split(split):
    """Process one split (train/val/test)"""
    print(f"\n🌦️  Augmenting {split} split...")
    
    img_dir = os.path.join(IMAGES_ROOT, split)
    label_dir = os.path.join(LABELS_ROOT, split)
    
    aug_img_dir = os.path.join(AUG_IMAGES_ROOT, split)
    aug_label_dir = os.path.join(AUG_LABELS_ROOT, split)
    
    # Create output directories
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_label_dir, exist_ok=True)
    
    # Get all images
    image_files = glob(os.path.join(img_dir, "*.jpg"))
    
    if not image_files:
        print(f"⚠️  No images found in {img_dir}")
        return
    
    augmentations = ['fog', 'rain', 'snow', 'lowlight']
    total = len(image_files) * len(augmentations)
    
    success = 0
    failed = 0
    
    with tqdm(total=total, desc=f"Processing {split}") as pbar:
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            
            # Find corresponding label
            label_path = os.path.join(label_dir, base_name + '.txt')
            
            # Apply each augmentation
            for aug_type in augmentations:
                try:
                    # Augment image
                    aug_img = augment_image(img_path, aug_type)
                    
                    if aug_img is None:
                        failed += 1
                        pbar.update(1)
                        continue
                    
                    # Save augmented image
                    aug_img_name = f"{base_name}_{aug_type}.jpg"
                    aug_img_path = os.path.join(aug_img_dir, aug_img_name)
                    cv2.imwrite(aug_img_path, aug_img)
                    
                    # Copy label if exists
                    if os.path.exists(label_path):
                        aug_label_name = f"{base_name}_{aug_type}.txt"
                        aug_label_path = os.path.join(aug_label_dir, aug_label_name)
                        shutil.copy(label_path, aug_label_path)
                    
                    success += 1
                    
                except Exception as e:
                    print(f"\n⚠️  Error processing {img_name} with {aug_type}: {e}")
                    failed += 1
                
                pbar.update(1)
    
    print(f"✅ {split}: Success={success}, Failed={failed}")

# Main execution
print("="*70)
print("🌦️  WEATHER AUGMENTATION PIPELINE")
print("="*70)
print("\nGenerating synthetic weather conditions:")
print("  - Fog")
print("  - Rain")
print("  - Snow")
print("  - Low-light/Night")
print("\n" + "="*70)

for split in ['train', 'val', 'test']:
    process_split(split)

print("\n" + "="*70)
print("✅ AUGMENTATION COMPLETE!")
print("="*70)
print(f"\n📊 Expected Results:")
print(f"   Train: ~280,000 images (70k × 4)")
print(f"   Val:   ~40,000 images (10k × 4)")
print(f"   Test:  ~80,000 images (20k × 4)")
print(f"\n📂 Augmented images saved in: {AUG_IMAGES_ROOT}")
print(f"📂 Augmented labels saved in: {AUG_LABELS_ROOT}")
print("\n🎯 Next step: Verify augmentation with verify.py")
