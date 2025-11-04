import os
import random
from glob import glob
from tqdm import tqdm
import shutil
import cv2
import numpy as np

# ------------- CONFIG -------------
BASE = "bdd100k"
IMAGES_ROOT = os.path.join(BASE, "images_100k")        # original images
YOLO_LABELS_ROOT = os.path.join(BASE, "yolo_format")  # existing YOLO labels
AUG_IMAGES_ROOT = os.path.join(BASE, "augmented")     # output augmented images
AUG_LABELS_ROOT = os.path.join(BASE, "augmented_labels") # output augmented labels
SPLITS = ["train", "val", "test"]
TEST_LIMIT = None   # set int (e.g., 200) for quick tests, or None for all
MAKE_ORIGINAL_LABEL_COPY = True  # also copy original label to augmented_labels

# ------------- Utilities -------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_images(split_img_dir):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(split_img_dir, e)))
    return sorted(files)

def copy_label(orig_basename, split, out_basename):
    src = os.path.join(YOLO_LABELS_ROOT, split, orig_basename + ".txt")
    dst_dir = os.path.join(AUG_LABELS_ROOT, split)
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, out_basename + ".txt")
    if os.path.exists(src):
        shutil.copy(src, dst)
        return True
    return False

# ---------------- EXTREME EFFECTS ----------------

def add_fog(img):
    """
    Strong fog with depth gradient and color tint.
    """
    h, w = img.shape[:2]
    # create a depth-like gradient (closer = less fog, far = more fog)
    grad = np.linspace(0, 1, h).reshape(h, 1)  # vertical depth gradient
    grad = cv2.resize(grad, (w, h))
    # fog strength more aggressive
    base_alpha = random.uniform(0.35, 0.75)  # stronger than before
    # add slight horizontal variation
    hor_variation = (np.random.rand(h, w) * 0.15).astype(np.float32)
    alpha = np.clip(base_alpha * grad + hor_variation, 0.0, 0.92)

    # fog color tint (warm or cool haze)
    if random.random() < 0.5:
        tint = np.array([220, 230, 255], dtype=np.float32)  # cool bluish fog
    else:
        tint = np.array([235, 225, 200], dtype=np.float32)  # warm fog

    fog_layer = np.ones_like(img, dtype=np.float32) * tint.reshape(1,1,3)
    img_f = img.astype(np.float32)
    alpha_3 = alpha[..., None]
    out = img_f * (1 - alpha_3) + fog_layer * alpha_3
    out = np.clip(out, 0, 255).astype(np.uint8)

    # subtle blur to simulate scattering
    ksize = random.choice([5,7,9])
    out = cv2.GaussianBlur(out, (ksize, ksize), 0)

    return out

def add_rain(img):
    """
    Heavy rain: many streaks + motion blur to create sheets of rain.
    """
    h, w = img.shape[:2]
    rain_layer = np.zeros((h, w, 3), dtype=np.uint8)

    # increase number of streaks significantly
    n_streaks = int((w * h) / 1500)  # denser than previous
    for _ in range(n_streaks):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        length = random.randint(int(h*0.02), int(h*0.12))  # longer streaks
        slant = random.randint(-10, 10)
        x2 = min(w-1, max(0, x + slant))
        y2 = min(h-1, y + length)
        thickness = random.choice([1,1,2])  # some thicker streaks
        color = (200 + random.randint(0,35),) * 3
        cv2.line(rain_layer, (x, y), (x2, y2), color, thickness)

    # blur and motion-blur (directional) to simulate falling sheets
    rain_gray = cv2.cvtColor(rain_layer, cv2.COLOR_BGR2GRAY)
    rain_blur = cv2.GaussianBlur(rain_gray, (3,3), 0)
    # directional blur by convolving with a vertical kernel
    k_len = random.choice([7, 11, 15])
    kernel = np.zeros((k_len, k_len))
    kernel[:, k_len//2] = np.ones(k_len)
    kernel = kernel / k_len
    rain_mb = cv2.filter2D(rain_blur, -1, kernel)

    rain_norm = cv2.normalize(rain_mb, None, 0, 255, cv2.NORM_MINMAX)
    rain_col = cv2.merge([rain_norm, rain_norm, rain_norm])

    # merge with original - heavier effect: increase weight of rain layer
    alpha = 0.6 if random.random() < 0.7 else 0.5
    out = cv2.addWeighted(img, 1 - alpha, rain_col.astype(np.uint8), alpha, 0)

    # slight contrast drop and blur
    out = cv2.GaussianBlur(out, (3,3), 0)
    out = cv2.addWeighted(out, 0.95, np.zeros_like(out), 0, -15)  # lower brightness a bit

    return out

def add_lowlight(img):
    """
    Very dark scenes + stronger vignette + bright headlight-like flares (random).
    """
    h, w = img.shape[:2]
    # stronger gamma darkening
    gamma = random.uniform(2.2, 3.2)  # more dark
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    dark = cv2.LUT(img, table)

    # strong vignette
    X_resultant_kernel = cv2.getGaussianKernel(w, w*0.6)
    Y_resultant_kernel = cv2.getGaussianKernel(h, h*0.6)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / np.linalg.norm(kernel)
    mask = mask / np.max(mask)
    # invert mask so edges are darker
    vignette = (mask[..., None] * 0.6 + 0.4)  # control center vs edge
    dark = (dark.astype(np.float32) * vignette).astype(np.uint8)

    # optionally add headlight-like bright blobs to simulate glare sources (cars)
    if random.random() < 0.8:
        n_flares = random.randint(1, 3)
        for _ in range(n_flares):
            cx = random.randint(int(w*0.1), int(w*0.9))
            cy = random.randint(int(h*0.2), int(h*0.9))
            rad = random.randint(int(min(w,h)*0.03), int(min(w,h)*0.12))
            overlay = dark.copy().astype(np.float32)
            cv2.circle(overlay, (cx, cy), rad, (255, 240, 220), -1)
            alpha = random.uniform(0.08, 0.22)
            dark = cv2.addWeighted(dark.astype(np.float32), 1.0, overlay, alpha, 0).astype(np.uint8)
            # add small bloom
            dark = cv2.GaussianBlur(dark, (rad//2*2+1, rad//2*2+1), 0)

    # final small blur
    dark = cv2.GaussianBlur(dark, (3,3), 0)
    return dark

def add_snow(img):
    """
    Heavy snow: dense and larger flakes, some bloom, overall whitened atmosphere.
    """
    h, w = img.shape[:2]
    snow_layer = np.zeros((h, w), dtype=np.uint8)

    # denser, larger flakes
    n_dots = int((w*h) / 3000)  # heavier than before
    for _ in range(n_dots):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        r = random.randint(0, 3)  # bigger flakes occasionally
        cv2.circle(snow_layer, (x, y), r, 255, -1)

    # blur snow for depth and some streaks
    snow_blur = cv2.GaussianBlur(snow_layer, (7,7), 0)

    # make noise-based alpha mask
    alpha = (snow_blur.astype(np.float32) / 255.0) * random.uniform(0.5, 0.95)
    alpha = np.clip(alpha[..., None], 0.0, 0.95)

    out = img.astype(np.float32) * (1 - alpha) + 255 * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    # occasional motion streaks for windy snow
    if random.random() < 0.5:
        k_len = random.choice([5,9])
        kernel = np.zeros((k_len, k_len))
        kernel[k_len//2, :] = np.ones(k_len)
        kernel = kernel / k_len
        out = cv2.filter2D(out, -1, kernel)

    # slight global brightness increase to simulate whitened scene
    out = cv2.addWeighted(out, 1.05, np.zeros_like(out), 0, 10)

    return out

EFFECTS = {
    "fog": add_fog,
    "rain": add_rain,
    "lowlight": add_lowlight,
    "snow": add_snow
}

# ------------- Main Augment Function -------------
def augment_split(split):
    split_img_dir = os.path.join(IMAGES_ROOT, split)
    out_img_dir = os.path.join(AUG_IMAGES_ROOT, split)
    out_lbl_dir = os.path.join(AUG_LABELS_ROOT, split)
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    img_files = list_images(split_img_dir)
    if TEST_LIMIT:
        img_files = img_files[:TEST_LIMIT]

    if not img_files:
        print(f"⚠️ No images found for split '{split}' in {split_img_dir}")
        return 0, 0, 0

    processed = 0
    augmented_saved = 0
    labels_copied = 0

    for img_path in tqdm(img_files, desc=f"Augment {split}", unit="img"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        basename = os.path.splitext(os.path.basename(img_path))[0]
        processed += 1

        # Optionally copy original label
        if MAKE_ORIGINAL_LABEL_COPY:
            if copy_label(basename, split, basename):
                labels_copied += 1

        for eff_name, eff_func in EFFECTS.items():
            try:
                aug_img = eff_func(img)
            except Exception as e:
                print(f"[WARN] effect {eff_name} failed for {basename}: {e}")
                continue

            out_name = f"{basename}_{eff_name}.jpg"
            out_path = os.path.join(out_img_dir, out_name)
            # strong JPEG quality to keep details (or adjust as needed)
            cv2.imwrite(out_path, aug_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            augmented_saved += 1

            # copy label for augmented
            if copy_label(basename, split, f"{basename}_{eff_name}"):
                labels_copied += 1

    return processed, augmented_saved, labels_copied

def run_all():
    ensure_dir(AUG_IMAGES_ROOT)
    ensure_dir(AUG_LABELS_ROOT)
    total_proc = total_aug = total_lbl = 0

    for sp in SPLITS:
        p, a, l = augment_split(sp)
        print(f"\nSplit {sp}: images processed={p}, augmented saved={a}, labels copied={l}")
        total_proc += p; total_aug += a; total_lbl += l

    print("\nALL DONE")
    print(f"Total processed: {total_proc}, total augmented images: {total_aug}, total labels copied: {total_lbl}")
    print("Augmented images dir:", AUG_IMAGES_ROOT)
    print("Augmented labels dir:", AUG_LABELS_ROOT)

if __name__ == "__main__":
    run_all()
