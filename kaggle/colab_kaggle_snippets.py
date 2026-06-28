"""
================================================================================
 Vehicle Damage Detection & Severity — COLAB (data pulled from Kaggle)
================================================================================
Each block below is ONE Colab cell. Run them top to bottom.

This version runs in Google Colab but downloads the 2.9 GB CarDD dataset
directly from Kaggle via the API (no slow Drive upload). You only upload the
small 21.6 MB bundle (best_model.pt + your hand-labeled crops).

ONE-TIME PREP — get a Kaggle API token:
  1. kaggle.com -> your avatar -> Settings -> "Create New Token"
     (downloads kaggle.json to your computer).
  2. Run Snippet 1, click "Choose Files", and upload that kaggle.json.

Then Snippet 2 uploads upload_to_kaggle.zip (it's in your repo at
kaggle/upload_to_kaggle.zip).

Set Colab Runtime > Change runtime type > T4 GPU before running.
================================================================================
"""


# === SNIPPET 1 — GPU check + install + Kaggle token =========================
import torch
print("CUDA available:", torch.cuda.is_available(), "|",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

get_ipython().system('pip -q install ultralytics==8.* kaggle >/dev/null')

import os
from google.colab import files
print("\nUpload your kaggle.json (Kaggle > Settings > Create New Token):")
up = files.upload()                     # pick kaggle.json
os.makedirs('/root/.kaggle', exist_ok=True)
os.replace('kaggle.json', '/root/.kaggle/kaggle.json')
os.chmod('/root/.kaggle/kaggle.json', 0o600)
print("Kaggle token installed.")


# === SNIPPET 2 — Upload your small bundle (crops + weights) =================
from google.colab import files
print("Upload kaggle/upload_to_kaggle.zip from your repo:")
up = files.upload()                     # pick upload_to_kaggle.zip
import zipfile
os.makedirs('/content/mydata', exist_ok=True)
with zipfile.ZipFile('upload_to_kaggle.zip') as z:
    z.extractall('/content/mydata')
print("Extracted to /content/mydata:", sorted(os.listdir('/content/mydata')))


# === SNIPPET 3 — Download the CarDD dataset from Kaggle =====================
# If this exact slug 404s, run:  !kaggle datasets list -s "cardd yolo"
# and replace DATASET_SLUG with the owner/name it prints.
DATASET_SLUG = 'cardd-with-yolo-annotations-images-labels'
os.makedirs('/content/cardd', exist_ok=True)
get_ipython().system(f'kaggle datasets download -d {DATASET_SLUG} -p /content/cardd --unzip')
# Locate the folder that actually contains train/images
CARDD = None
for root, dirs, _ in os.walk('/content/cardd'):
    if 'train' in dirs and os.path.isdir(os.path.join(root, 'train', 'images')):
        CARDD = root
        break
print("CARDD root:", CARDD)
print("train images:", len(os.listdir(os.path.join(CARDD, 'train/images'))))


# === SNIPPET 4 — Set working paths + write corrected data.yaml ==============
WORK = '/content/work'
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
HAND = '/content/mydata'                              # minor/ moderate/ severe/
BEST = os.path.join(HAND, 'best_model.pt')
print("Detector weights:", BEST, "exists:", os.path.exists(BEST))
for c in ['minor', 'moderate', 'severe']:
    p = os.path.join(HAND, c)
    print(f"  {c}: {len(os.listdir(p)) if os.path.isdir(p) else 'MISSING'}")

DATA_YAML = os.path.join(WORK, 'data.yaml')
with open(DATA_YAML, 'w') as f:
    f.write(f"""path: {CARDD}
train: train/images
val: val/images
test: test/images
nc: 6
names:
  0: dent
  1: scratch
  2: crack
  3: glass shatter
  4: lamp broken
  5: tire flat
""")
print(open(DATA_YAML).read())


# === SNIPPET 5 — Validate the recovered detector ============================
# If this prints GOOD, SKIP snippet 6. If WEAK, run snippet 6.
from ultralytics import YOLO

detector = None
try:
    detector = YOLO(BEST)
    print("Loaded best_model.pt. Validating on the test split...")
    m = detector.val(data=DATA_YAML, split='test', verbose=False)
    print(f"\nRecovered weights — mAP@50={m.box.map50:.3f}  mAP@50-95={m.box.map:.3f}  "
          f"P={m.box.mp:.3f}  R={m.box.mr:.3f}")
    GOOD = m.box.map50 > 0.40
    print("\n>>> Weights look GOOD — skip snippet 6." if GOOD
          else "\n>>> Weights look WEAK — run snippet 6.")
except Exception as e:
    print("Could not validate recovered weights:", e)
    print(">>> Run snippet 6.")
    GOOD = False


# === SNIPPET 6 — Retrain detector (ONLY if snippet 5 said WEAK; ~30 min) =====
import shutil
model = YOLO('yolov8n.pt')
model.train(
    data=DATA_YAML, epochs=30, imgsz=640, batch=16,
    device=0 if torch.cuda.is_available() else 'cpu',
    patience=10, project=WORK + '/runs', name='cardd_detector', exist_ok=True,
)
m = model.val(data=DATA_YAML, split='test', verbose=False)
print(f"Retrained — mAP@50={m.box.map50:.3f}  mAP@50-95={m.box.map:.3f}")
BEST = WORK + '/runs/cardd_detector/weights/best.pt'
shutil.copy(BEST, WORK + '/best_model.pt')
detector = YOLO(BEST)
print("Saved new best_model.pt")


# === SNIPPET 7 — Severity labeling rule ======================================
import cv2, numpy as np
from pathlib import Path

DAMAGE_NAMES = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat']
SEVERE_TYPES = {'glass shatter', 'lamp broken', 'tire flat', 'crack'}

def rule_severity(damage, box_frac):
    if damage in SEVERE_TYPES:
        return 'severe' if box_frac > 0.05 else 'moderate'
    if damage == 'scratch':
        return 'moderate' if box_frac > 0.10 else 'minor'
    if damage == 'dent':
        if box_frac > 0.12: return 'severe'
        if box_frac > 0.04: return 'moderate'
        return 'minor'
    return 'moderate'


# === SNIPPET 8 — Auto-label extra severity crops (weak supervision) ==========
OUT = Path(WORK) / 'severity_auto'
for c in ['minor', 'moderate', 'severe']:
    (OUT / c).mkdir(parents=True, exist_ok=True)
counts = {'minor': 0, 'moderate': 0, 'severe': 0}
CONF = 0.30
SRC_DIRS = [os.path.join(CARDD, 'train/images'), os.path.join(CARDD, 'val/images')]

n_img = 0
for sd in SRC_DIRS:
    if not os.path.isdir(sd):
        continue
    for name in os.listdir(sd):
        img = cv2.imread(os.path.join(sd, name))
        if img is None:
            continue
        H, W = img.shape[:2]
        n_img += 1
        for r in detector(img, conf=CONF, verbose=False):
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                if x2 <= x1 or y2 <= y1:
                    continue
                damage = DAMAGE_NAMES[int(b.cls[0])] if int(b.cls[0]) < len(DAMAGE_NAMES) else 'dent'
                frac = ((x2 - x1) * (y2 - y1)) / float(W * H)
                sev = rule_severity(damage, frac)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                counts[sev] += 1
                cv2.imwrite(str(OUT / sev / f"auto_{counts[sev]}.jpg"), crop)
print(f"Scanned {n_img} images. Auto-labeled crops:", counts, "total", sum(counts.values()))


# === SNIPPET 9 — Build train (auto + hand) and val (hand only) splits ========
import random
CLASSES = ['minor', 'moderate', 'severe']
train_root = Path(WORK) / 'sev_train'
val_root = Path(WORK) / 'sev_val'
for root in (train_root, val_root):
    if root.exists():
        shutil.rmtree(root)
    for c in CLASSES:
        (root / c).mkdir(parents=True, exist_ok=True)

random.seed(42)
for c in CLASSES:
    src = Path(HAND) / c
    files_ = [f for f in src.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')] if src.is_dir() else []
    random.shuffle(files_)
    k = max(1, int(len(files_) * 0.2))
    for f in files_[:k]:
        shutil.copy(f, val_root / c / f.name)               # trusted val
    for f in files_[k:]:
        shutil.copy(f, train_root / c / ('hand_' + f.name))   # hand -> train
for c in CLASSES:
    ad = OUT / c
    if ad.is_dir():
        for f in ad.iterdir():
            shutil.copy(f, train_root / c / f.name)          # auto -> train
for c in CLASSES:
    print(f"{c:9s} train={len(list((train_root/c).iterdir())):4d}  val={len(list((val_root/c).iterdir())):3d}")


# === SNIPPET 10 — Train the severity classifier (MobileNetV2) ================
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG, BATCH = (224, 224), 16
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_root, image_size=IMG, batch_size=BATCH, class_names=CLASSES, label_mode='int', seed=42)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_root, image_size=IMG, batch_size=BATCH, class_names=CLASSES, label_mode='int', shuffle=False)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(500).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

aug = models.Sequential([layers.RandomFlip('horizontal'),
                         layers.RandomRotation(0.1), layers.RandomZoom(0.1)])
base = MobileNetV2(input_shape=IMG + (3,), include_top=False, weights='imagenet')
base.trainable = False
model = models.Sequential([
    layers.Rescaling(1 / 127.5, offset=-1), aug, base,
    layers.GlobalAveragePooling2D(), layers.Dropout(0.3),
    layers.Dense(len(CLASSES), activation='softmax')])
model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(train_ds, validation_data=val_ds, epochs=25,
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
print("\nBest val accuracy:", round(max(hist.history['val_accuracy']), 3))
model.save(WORK + '/severity_model.h5')
open(WORK + '/severity_classes.txt', 'w').write("\n".join(CLASSES))
print("Saved severity_model.h5")


# === SNIPPET 11 — Confusion matrix on the trusted hand-labeled val set =======
y_true, y_pred = [], []
for x, y in val_ds:
    p = model.predict(x, verbose=0)
    y_true += list(y.numpy())
    y_pred += list(p.argmax(1))
cm = np.zeros((3, 3), int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1
print("rows=true, cols=pred  order:", CLASSES)
print(cm)
print("val accuracy (hand-labeled):", round(float(np.trace(cm) / cm.sum()) if cm.sum() else 0, 3))


# === SNIPPET 12 — End-to-end demo ============================================
import glob
from IPython.display import Image as IPyImage, display

COLORS = {'minor': (0, 200, 0), 'moderate': (0, 165, 255), 'severe': (0, 0, 255)}
SEV = open(WORK + '/severity_classes.txt').read().split()

def assess(image_path, out):
    img = cv2.imread(image_path)
    assert img is not None, image_path
    found = []
    for r in detector(img, conf=0.30, verbose=False):
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            if x2 <= x1 or y2 <= y1:
                continue
            dmg = DAMAGE_NAMES[int(b.cls[0])]
            crop = cv2.resize(img[y1:y2, x1:x2], (224, 224)).astype('float32')
            sev = SEV[int(model.predict(crop[None], verbose=0)[0].argmax())]
            col = COLORS.get(sev, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f"{dmg} | {sev}", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            found.append((dmg, sev))
    cv2.imwrite(out, img)
    print(f"{os.path.basename(image_path)}: {len(found)} regions ->", found)
    return out

tests = sorted(glob.glob(os.path.join(CARDD, 'test/images/*.jpg')))[:3]
for i, t in enumerate(tests):
    o = assess(t, WORK + f'/demo_{i}.jpg')
    display(IPyImage(o))

# Download demo_0.jpg + severity_model.h5 + best_model.pt from the Files panel
# (left sidebar, /content/work), then send me your metrics to finish the README.
