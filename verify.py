import os
from glob import glob

splits = ["train", "val", "test"]
base_img = "bdd100k/augmented"
base_lbl = "bdd100k/augmented_labels"

for sp in splits:
    imgs = glob(os.path.join(base_img, sp, "*.jpg"))
    lbls = glob(os.path.join(base_lbl, sp, "*.txt"))
    print(f"{sp}: {len(imgs)} images, {len(lbls)} labels")
