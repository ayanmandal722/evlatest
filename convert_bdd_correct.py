# convert_bdd_correct.py
import os
import json
from glob import glob
from PIL import Image
from tqdm import tqdm

BASE = "bdd100k"
IMAGES_ROOT = os.path.join(BASE, "images", "100k")
LABELS_ROOT = os.path.join(BASE, "labels", "det_20")
OUTPUT_ROOT = os.path.join(BASE, "yolo_format")

BDD_CLASSES = [
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign"
]

def convert_bbox_to_yolo(img_width, img_height, box):
    """Convert BDD100K bbox to YOLO format"""
    x1, y1 = box['x1'], box['y1']
    x2, y2 = box['x2'], box['y2']
    
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return max(0, min(1, x_center)), max(0, min(1, y_center)), max(0, min(1, width)), max(0, min(1, height))

def process_split(split):
    print(f"\n{'='*60}")
    print(f"Processing {split.upper()} split")
    print('='*60)
    
    img_dir = os.path.join(IMAGES_ROOT, split)
    label_dir = os.path.join(LABELS_ROOT, split)
    output_dir = os.path.join(OUTPUT_ROOT, split)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build image map
    print("📂 Building image name map...")
    image_files = glob(os.path.join(img_dir, "*.jpg"))
    image_map = {}
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        image_map[base_name] = img_path
        image_map[img_name] = img_path
    
    print(f"✅ Found {len(image_files):,} images")
    
    # Get JSON files
    json_files = sorted(glob(os.path.join(label_dir, "*.json")))
    
    if not json_files:
        print(f"❌ No JSON files found")
        return 0, 0
    
    print(f"📂 Found {len(json_files):,} JSON files")
    
    converted = 0
    skipped = 0
    
    for json_path in tqdm(json_files, desc=f"Converting {split}"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get image name
            img_name = data.get('name', '')
            
            if not img_name:
                skipped += 1
                continue
            
            # Find image
            img_path = image_map.get(img_name) or image_map.get(img_name + '.jpg')
            
            if not img_path or not os.path.exists(img_path):
                skipped += 1
                continue
            
            # Get image dimensions
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            # Create label file
            label_name = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
            label_path = os.path.join(output_dir, label_name)
            
            labels_written = 0
            
            # KEY FIX: Extract objects from frames[0]["objects"]
            objects = []
            if 'frames' in data and len(data['frames']) > 0:
                if 'objects' in data['frames'][0]:
                    objects = data['frames'][0]['objects']
            # Fallback: check if labels key exists (old format)
            elif 'labels' in data:
                objects = data['labels']
            
            with open(label_path, 'w') as f_out:
                for obj in objects:
                    category = obj.get('category', '')
                    
                    if category not in BDD_CLASSES:
                        continue
                    
                    if 'box2d' not in obj:
                        continue
                    
                    class_id = BDD_CLASSES.index(category)
                    x_c, y_c, w, h = convert_bbox_to_yolo(img_width, img_height, obj['box2d'])
                    
                    f_out.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                    labels_written += 1
            
            if labels_written > 0:
                converted += 1
            else:
                if os.path.exists(label_path):
                    os.remove(label_path)
                skipped += 1
        
        except Exception as e:
            skipped += 1
    
    print(f"\n📊 Results:")
    print(f"   ✅ Converted: {converted:,}")
    print(f"   ⚠️  Skipped: {skipped:,}")
    
    return converted, skipped

def main():
    print("\n" + "="*60)
    print("🔧 BDD100K to YOLO Conversion (CORRECT FORMAT)")
    print("="*60)
    
    total_converted = 0
    
    for split in ['train', 'val', 'test']:
        converted, skipped = process_split(split)
        total_converted += converted
    
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE!")
    print("="*60)
    print(f"\nTotal converted: {total_converted:,}")
    
    # Verify
    print("\n🔍 Verification:")
    for split in ['train', 'val', 'test']:
        txt_count = len(glob(os.path.join(OUTPUT_ROOT, split, '*.txt')))
        print(f"   {split}: {txt_count:,} labels")

if __name__ == "__main__":
    main()

