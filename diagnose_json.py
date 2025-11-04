# diagnose_json.py
import json
import os
from glob import glob

BASE = "bdd100k"
LABELS_ROOT = os.path.join(BASE, "labels", "det_20", "train")

# Get first 3 JSON files
json_files = sorted(glob(os.path.join(LABELS_ROOT, "*.json")))[:3]

print("="*60)
print("🔍 DIAGNOSING JSON FORMAT")
print("="*60)

for i, json_path in enumerate(json_files, 1):
    print(f"\n📄 File {i}: {os.path.basename(json_path)}")
    print("-"*60)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Keys in JSON: {list(data.keys())}")
        
        if 'name' in data:
            print(f"Image name: {data['name']}")
        
        if 'labels' in data:
            print(f"Number of labels: {len(data['labels'])}")
            
            if len(data['labels']) > 0:
                label = data['labels'][0]
                print(f"\nFirst label structure:")
                print(f"  Keys: {list(label.keys())}")
                
                if 'category' in label:
                    print(f"  Category: {label['category']}")
                
                if 'box2d' in label:
                    print(f"  Box2d: {label['box2d']}")
                elif 'bbox' in label:
                    print(f"  Bbox: {label['bbox']}")
                
                # Show all detection labels
                detection_labels = [l for l in data['labels'] if 'box2d' in l or 'bbox' in l]
                print(f"\nTotal detection labels in this file: {len(detection_labels)}")
                
                if len(detection_labels) > 0:
                    categories = [l.get('category', 'unknown') for l in detection_labels]
                    print(f"Categories found: {set(categories)}")
        
        print("\n" + "="*60)
        print("FULL JSON CONTENT (first file only):")
        if i == 1:
            print(json.dumps(data, indent=2)[:1000])  # Print first 1000 chars
            print("...")
    
    except Exception as e:
        print(f"ERROR reading file: {e}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
