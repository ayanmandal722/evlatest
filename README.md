```markdown
# 🚗 BDD100K Weather Augmentation & YOLO Preprocessing

## 📌 Project Overview
This project is part of our college work to explore **object detection in adverse weather**.  
We use the **BDD100K dataset (10K subset)**, convert its annotations to **YOLO format**,  
generate **synthetic weather variations** (fog, rain, snow, low-light),  
and produce a **comparative summary** of datasets.

---

## 🚀 Features
- ✅ JSON → YOLO label conversion (`convert_bdd_to_yolo.py`)  
- ✅ Weather augmentation (`augment_dataset.py`)  
- ✅ Dataset statistics & charts (`generate_summary_md.py`)  
- ✅ Diagnostic tools (`diagnose_labels.py`, `debug_plot_train.py`)  
- ✅ Comparative report (`report/comparative_summary.md` + charts)  

---

## 📂 Folder Structure
```

bdd100k/
images\_10k/          # Original dataset images (NOT uploaded)
labels/              # Original JSON annotations (NOT uploaded)
yolo\_format/         # YOLO .txt labels (generated locally)
augmented/           # Augmented images (NOT uploaded)
augmented\_labels/    # Augmented YOLO labels (generated locally)

report/
comparative\_summary.md
images/              # Charts + sample images

````

---

## ⚠️ Dataset
The datasets are **too large for GitHub**, so they are **not included** here.  
👉 Download from the official site: [BDD100K dataset](https://bdd-data.berkeley.edu/)

---

## 🛠️ How to Run

### 1. Convert JSON → YOLO
```bash
python convert_bdd_to_yolo.py
````

### 2. Augment dataset (fog, rain, low-light, snow)

```bash
python augment_dataset.py
```

### 3. Generate dataset summary & charts

```bash
python generate_summary_md.py
```

This creates:

* `report/comparative_summary.md`
* Charts & sample images in `report/images/`

---

## 📊 Deliverables (Part 1)

* ✅ Curated dataset (local only)
* ✅ Augmentation pipeline
* ✅ Comparative report (`report/comparative_summary.md`)

---

## 📌 Next Steps

* Part 2 → Model development (YOLO baseline + improvements)
* Part 3 → Simulation testing (CARLA or equivalent)

---

## 👨‍💻 Contributors

* Mohit Agarwal
* Moubani
* Ayan
* Subhojit
* Swatadru


```





