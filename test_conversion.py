# test_conversion.py - TEST ON 100 FILES ONLY
import os
import json
from glob import glob
from PIL import Image
from tqdm import tqdm

BASE = "bdd100k"
IMAGES_ROOT = os.path.join(BASE, "images", "100k", "train")
LABELS_ROOT = os.path.join(BASE, "labels", "det_20", "train")
OUTPUT_ROOT = os.path.join(BASE, "yolo_format_test")  # Different folder for testing

BDD_CLASSES = [
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign"
]

def convert_bbox_to_yolo(img_width, img_height, box):
    x1, y1 = box['x1'], box['y1']
    x2, y2 = box['x2'], box['y2']
    
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return max(0, min(1, x_center)), max(0, min(1, y_center)), max(0, min(1, width)), max(0, min(1, height))

print("="*60)
print("🧪 TESTING CONVERSION ON 100 FILES")
print("="*60)

# Build image map
print("\n📂 Building image map...")
image_files = glob(os.path.join(IMAGES_ROOT, "*.jpg"))
image_map = {}

for img_path in image_files:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    image_map[base_name] = img_path

print(f"✅ Found {len(image_files):,} total images")

# Get ONLY 100 JSON files for testing
json_files = sorted(glob(os.path.join(LABELS_ROOT, "*.json")))[:100]

print(f"🧪 Testing on {len(json_files)} JSON files\n")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

converted = 0
skipped = 0
details = []

for json_path in tqdm(json_files, desc="Testing"):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_name = data.get('name', '')
        
        if not img_name:
            skipped += 1
            details.append(f"❌ {os.path.basename(json_path)}: No name in JSON")
            continue
        
        img_path = image_map.get(img_name)
        
        if not img_path:
            skipped += 1
            details.append(f"❌ {img_name}: Image not found")
            continue
        
        with Image.open(img_path) as img:
            img_width, img_height = img.size
        
        label_name = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        label_path = os.path.join(OUTPUT_ROOT, label_name)
        
        labels_written = 0
        
        # Extract objects from frames
        objects = []
        if 'frames' in data and len(data['frames']) > 0:
            if 'objects' in data['frames'][0]:
                objects = data['frames'][0]['objects']
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
            details.append(f"✅ {img_name}: {labels_written} objects")
        else:
            if os.path.exists(label_path):
                os.remove(label_path)
            skipped += 1
            details.append(f"⚠️ {img_name}: No valid objects")
    
    except Exception as e:
        skipped += 1
        details.append(f"❌ {os.path.basename(json_path)}: Error - {str(e)[:50]}")

print("\n" + "="*60)
print("📊 TEST RESULTS")
print("="*60)
print(f"✅ Successfully converted: {converted}/{len(json_files)}")
print(f"⚠️  Skipped: {skipped}/{len(json_files)}")
print(f"Success rate: {converted/len(json_files)*100:.1f}%")

# Show first 10 details
print("\n📋 Sample Results (first 10):")
for detail in details[:10]:
    print(f"  {detail}")

if converted > 0:
    print("\n🎉 SUCCESS! The conversion works!")
    print(f"✅ {converted} files converted successfully")
    print(f"\n📁 Test labels saved in: {OUTPUT_ROOT}")
    
    # Show a sample label file
    sample_label = glob(os.path.join(OUTPUT_ROOT, "*.txt"))[0]
    print(f"\n📄 Sample label file: {os.path.basename(sample_label)}")
    with open(sample_label, 'r') as f:
        lines = f.readlines()[:3]
        print("   First 3 lines:")
        for line in lines:
            print(f"   {line.strip()}")
    
    print("\n✅ Ready to run on full dataset!")
    print("   Run: python convert_bdd_correct.py")
else:
    print("\n❌ FAILED! No files converted")
    print("Check the details above for issues")

print("="*60)
