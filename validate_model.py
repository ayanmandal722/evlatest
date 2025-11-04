# validate_model.py
from ultralytics import YOLO
import os

print("="*70)
print("🔍 Model Validation on Test Set")
print("="*70)

# Find the best model
best_model = 'runs/train/bdd100k_weather_augmented/weights/best.pt'

if not os.path.exists(best_model):
    print(f"❌ Model not found: {best_model}")
    print("   Train the model first!")
    exit(1)

print(f"\n📦 Loading model: {best_model}")

# Load the trained model
model = YOLO(best_model)

print("\n🧪 Running validation on test set...")

# Validate
results = model.val(
    data='bdd100k_augmented.yaml',
    split='test',
    batch=16,
    imgsz=640,
    device=0,
    save_json=True,
    plots=True,
)

print("\n" + "="*70)
print("✅ VALIDATION COMPLETE!")
print("="*70)
print(f"\n📊 Results:")
print(f"   mAP50: {results.box.map50:.4f}")
print(f"   mAP50-95: {results.box.map:.4f}")
print(f"   Precision: {results.box.mp:.4f}")
print(f"   Recall: {results.box.mr:.4f}")

print("\n📁 Results saved in: runs/val/")

