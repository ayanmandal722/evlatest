import os
from glob import glob
from collections import Counter
import matplotlib
matplotlib.use("Agg")   # ensure non-GUI backend
import matplotlib.pyplot as plt

BASE = "bdd100k"
YOLO_DIR = os.path.join(BASE, "yolo_format", "train")  # use your YOLO folder
CLASSES = [
    "person","rider","car","truck","bus","train","motorcycle","bicycle",
    "traffic light","traffic sign"
]

def parse_yolo_labels(folder):
    c = Counter()
    if not os.path.isdir(folder):
        print("Folder not found:", folder)
        return c
    files = glob(os.path.join(folder, "*.txt"))
    print("Label files found:", len(files))
    for f in files:
        with open(f, "r", encoding="utf-8") as fo:
            for line in fo:
                parts = line.strip().split()
                if not parts: 
                    continue
                try:
                    cls = int(parts[0])
                except:
                    continue
                if 0 <= cls < len(CLASSES):
                    c[CLASSES[cls]] += 1
    return c

counts = parse_yolo_labels(YOLO_DIR)
print("Counts:", counts)

# If no counts, print a few label files for inspection
if sum(counts.values()) == 0:
    sample = glob(os.path.join(YOLO_DIR, "*.txt"))[:5]
    print("Sample label files (first 5):", sample)
    for s in sample:
        print("----", s)
        with open(s, "r", encoding="utf-8") as fo:
            print(fo.read())

# Plot if we have some data (even zeros will produce plot)
labels = CLASSES
values = [counts.get(l, 0) for l in labels]
plt.figure(figsize=(10,4))
plt.bar(labels, values)
plt.xticks(rotation=45, ha='right')
plt.title("Train - Class counts (YOLO labels)")
plt.tight_layout()
out = os.path.join("report", "images", "debug_train_counts.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150)
print("Saved chart to", out)
