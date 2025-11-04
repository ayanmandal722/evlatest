# train_yolo.py
from ultralytics import YOLO
import os

print("="*70)
print("🚀 BDD100K YOLO Training - Weather Augmented Dataset")
print("="*70)

# Check if dataset YAML exists
if not os.path.exists('bdd100k_augmented.yaml'):
    print("❌ ERROR: bdd100k_augmented.yaml not found!")
    print("   Please create the YAML file first.")
    exit(1)

print("\n📊 Dataset Configuration:")
print("   - Train: 280,000 augmented images")
print("   - Val: 40,000 augmented images")
print("   - Test: 80,000 augmented images")
print("   - Classes: 10 (person, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign)")
print("   - Augmentations: fog, rain, snow, lowlight")

# Training configuration
MODEL = 'yolov8n.pt'  # Nano model (fastest)
EPOCHS = 5  # You can increase this later
BATCH_SIZE = 16  # Adjust based on your GPU memory
IMG_SIZE = 640

print(f"\n⚙️  Training Configuration:")
print(f"   - Model: {MODEL}")
print(f"   - Epochs: {EPOCHS}")
print(f"   - Batch Size: {BATCH_SIZE}")
print(f"   - Image Size: {IMG_SIZE}")

# Ask for confirmation
print("\n" + "="*70)
response = input("🚀 Ready to start training? This will take several hours. (y/n): ")

if response.lower() != 'y':
    print("❌ Training cancelled.")
    exit(0)

print("\n" + "="*70)
print("🏋️  STARTING TRAINING...")
print("="*70)

# Load model
model = YOLO(MODEL)

# Train the model
results = model.train(
    data='bdd100k_augmented.yaml',
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    device=0,  # Use GPU 0 (change to 'cpu' if no GPU)
    workers=4,
    project='runs/train',
    name='bdd100k_weather_augmented',
    exist_ok=True,
    pretrained=True,
    optimizer='Adam',
    verbose=True,
    save=True,
    save_period=5,  # Save checkpoint every 5 epochs
    patience=10,  # Early stopping patience
    plots=True,
    # Data augmentation (additional to our weather augmentation)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"\n📊 Training Results:")
print(f"   Best model saved at: runs/train/bdd100k_weather_augmented/weights/best.pt")
print(f"   Last model saved at: runs/train/bdd100k_weather_augmented/weights/last.pt")

print("\n📈 View results:")
print("   - Training plots: runs/train/bdd100k_weather_augmented/")
print("   - Validation results: Check results.png and results.csv")

print("\n🔍 Next steps:")
print("   1. Validate model: python validate_model.py")
print("   2. Test on images: python test_model.py")
print("   3. Export model: python export_model.py")
