import os, glob, matplotlib.pyplot as plt

CLASSES = ["person", "car"]

counts = {c:0 for c in CLASSES}
labels = glob.glob("yolo_labels/*.txt")

print("Total images with labels:", len(labels))

for lf in labels:
    for line in open(lf):
        cls = int(line.split()[0])
        counts[CLASSES[cls]] += 1

print("Counts:", counts)

plt.bar(counts.keys(), counts.values())
plt.title("Class distribution")
plt.savefig("class_counts.png")
print("Saved chart as class_counts.png")