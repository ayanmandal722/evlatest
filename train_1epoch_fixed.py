# train_1epoch_fixed.py
from ultralytics import YOLO
import time

def main():
    print("="*70)
    print("🧪 1-EPOCH TEST - WINDOWS FIXED VERSION")
    print("="*70)
    
    print("\n✅ Labels verified:")
    print("   - Train: 280,000 images + 280,000 labels")
    print("   - Val: 40,000 images + 40,000 labels")
    print("   - Test: 80,000 images + 80,000 labels")
    
    print("\n⚙️  Configuration:")
    print("   - Model: YOLOv8n (Nano)")
    print("   - Epochs: 1")
    print("   - Batch Size: 8")
    print("   - Workers: 0 (Windows fix)")
    print("   - GPU: CUDA enabled")
    print("\n⏱️  Expected time: 3-4 hours")
    
    response = input("\n🚀 Start training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    print("\n" + "="*70)
    print("🏋️  STARTING TRAINING...")
    print("="*70)
    
    start_time = time.time()
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Train with Windows-compatible settings
    try:
        results = model.train(
            data='bdd100k_augmented.yaml',
            epochs=1,
            batch=8,
            imgsz=640,
            device=0,
            workers=0,  # CRITICAL: Must be 0 on Windows!
            project='runs/train',
            name='test_1epoch_final',
            exist_ok=True,
            verbose=True,
            save=True,
            amp=False,  # Disabled for GTX 1650
            plots=True,
        )
        
        elapsed_hours = (time.time() - start_time) / 3600
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\n⏱️  Actual Time: {elapsed_hours:.2f} hours")
        print(f"\n📊 Time Estimates for Full Training:")
        print(f"   - 10 epochs: {elapsed_hours * 10:.1f} hours ({elapsed_hours * 10 / 24:.1f} days)")
        print(f"   - 15 epochs: {elapsed_hours * 15:.1f} hours ({elapsed_hours * 15 / 24:.1f} days) ✅ Recommended")
        print(f"   - 20 epochs: {elapsed_hours * 20:.1f} hours ({elapsed_hours * 20 / 24:.1f} days)")
        
        print(f"\n📁 Results:")
        print(f"   - Best model: runs/train/test_1epoch_final/weights/best.pt")
        print(f"   - Last model: runs/train/test_1epoch_final/weights/last.pt")
        print(f"   - Plots: runs/train/test_1epoch_final/")
        
        print("\n✅ Next: Review results and run full training with 15 epochs!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        elapsed_hours = (time.time() - start_time) / 3600
        print(f"Ran for {elapsed_hours:.2f} hours before error")

if __name__ == '__main__':
    main()
