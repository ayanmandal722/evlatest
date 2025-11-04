# test_1epoch.py - Quick 1 epoch test for GTX 1650
from ultralytics import YOLO
import time
import os

print("="*70)
print("🧪 1-EPOCH TEST - GTX 1650")
print("="*70)

# Check if YAML exists
if not os.path.exists('bdd100k_augmented.yaml'):
    print("❌ ERROR: bdd100k_augmented.yaml not found!")
    print("   Please create the YAML file first.")
    exit(1)

# Configuration for testing
MODEL = 'yolov8n.pt'
EPOCHS = 1  # Just 1 epoch for testing
BATCH_SIZE = 8  # Safe for GTX 1650
IMG_SIZE = 640
WORKERS = 2

print(f"\n⚙️  Test Configuration:")
print(f"   - Model: {MODEL} (YOLOv8 Nano)")
print(f"   - Epochs: {EPOCHS}")
print(f"   - Batch Size: {BATCH_SIZE}")
print(f"   - Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"   - GPU: GTX 1650")
print(f"\n⏱️  Expected Time: 3-4 hours")

print("\n📊 This test will:")
print("   ✅ Verify dataset loads correctly")
print("   ✅ Check GPU is working")
print("   ✅ Measure actual training speed")
print("   ✅ Create a basic model")

response = input("\n🚀 Start 1-epoch test? (y/n): ")
if response.lower() != 'y':
    print("❌ Test cancelled.")
    exit(0)

print("\n" + "="*70)
print("🏋️  STARTING 1-EPOCH TEST...")
print("="*70)

# Record start time
start_time = time.time()

# Load model
print("\n📦 Loading YOLOv8n model...")
model = YOLO(MODEL)

# Train for 1 epoch
try:
    results = model.train(
        data='bdd100k_augmented.yaml',
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=0,  # Use GPU
        workers=WORKERS,
        project='runs/train',
        name='test_1epoch',
        exist_ok=True,
        verbose=True,
        save=True,
        amp=True,  # Mixed precision
        plots=True,
    )
    
    # Calculate time
    end_time = time.time()
    elapsed_hours = (end_time - start_time) / 3600
    
    print("\n" + "="*70)
    print("✅ 1-EPOCH TEST COMPLETE!")
    print("="*70)
    
    print(f"\n⏱️  Actual Time: {elapsed_hours:.2f} hours")
    print(f"\n📊 Estimated times for full training:")
    print(f"   - 10 epochs: {elapsed_hours * 10:.1f} hours ({elapsed_hours * 10 / 24:.1f} days)")
    print(f"   - 15 epochs: {elapsed_hours * 15:.1f} hours ({elapsed_hours * 15 / 24:.1f} days) ✅ Recommended")
    print(f"   - 20 epochs: {elapsed_hours * 20:.1f} hours ({elapsed_hours * 20 / 24:.1f} days)")
    
    print(f"\n📁 Results saved in: runs/train/test_1epoch/")
    print(f"   - Model: runs/train/test_1epoch/weights/best.pt")
    print(f"   - Plots: runs/train/test_1epoch/results.png")
    
    print("\n✅ Test successful! Ready for full training.")
    print("   Edit EPOCHS = 15 and run again for full training.")

except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
    end_time = time.time()
    elapsed_hours = (end_time - start_time) / 3600
    print(f"   Ran for {elapsed_hours:.2f} hours before stopping")
    
except Exception as e:
    print(f"\n\n❌ ERROR during training:")
    print(f"   {str(e)}")
    print("\n🔍 Common issues:")
    print("   1. Check if YAML file paths are correct")
    print("   2. Verify GPU is available: nvidia-smi")
    print("   3. Check if augmented images exist")
    
finally:
    print("\n" + "="*70)
