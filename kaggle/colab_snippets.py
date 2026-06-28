"""
================================================================================
 Vehicle Damage Detection & Severity — Colab snippets
================================================================================
Each block below is ONE Colab cell. Copy a block (between the "# === SNIPPET N"
banners) into its own cell and run them top to bottom.

This is the same flow as kaggle/colab_pipeline.ipynb, just split so you can run
and inspect one step at a time.

BEFORE YOU START — upload a folder to Google Drive (e.g. MyDrive/auto_damage/)
containing:
    best_model.pt   data.yaml   train/   val/   test/   minor/   moderate/   severe/
Then set PROJECT_DIR in Snippet 3.
================================================================================
"""


# === SNIPPET 1 — Check GPU ===================================================
# Runtime > Change runtime type > T4 GPU first. Expect "CUDA available: True".
import torch
print("CUDA available:", torch.cuda.is_available(), "|",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")


# === SNIPPET 2 — Install Ultralytics =========================================
get_ipython().system('pip -q install ultralytics==8.* >/dev/null')
import ultralytics
print("ultralytics", ultralytics.__version__)


# === SNIPPET 3 — Mount Drive and set the project folder ======================
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = '/content/drive/MyDrive/auto_damage'   # <-- EDIT to your folder

import os
os.chdir(PROJECT_DIR)
print("Working dir:", os.getcwd())
print("Contents:", sorted(os.listdir('.')))
# The list must include: best_model.pt data.yaml train val test minor moderate severe


# === SNIPPET 4 — Validate the recovered detector ============================
# If this prints GOOD, SKIP snippet 5. If WEAK, run snippet 5.
from ultralytics import YOLO

DATA_YAML = 'data.yaml'
detector = None
try:
    detector = YOLO('best_model.pt')
    print("Loaded best_model.pt. Validating on the test split...")
    m = detector.val(data=DATA_YAML, split='test', verbose=False)
    print(f"\nRecovered weights — mAP@50={m.box.map50:.3f}  mAP@50-95={m.box.map:.3f}  "
          f"P={m.box.mp:.3f}  R={m.box.mr:.3f}")
    GOOD = m.box.map50 > 0.40
    print("\n>>> Weights look GOOD — skip snippet 5." if GOOD
          else "\n>>> Weights look WEAK — run snippet 5.")
except Exception as e:
    print("Could not validate recovered weights:", e)
    print(">>> Run snippet 5.")
    GOOD = False


# === SNIPPET 5 — Retrain detector (ONLY if snippet 4 said WEAK; ~30 min) =====
model = YOLO('yolov8n.pt')
results = model.train(
    data=DATA_YAML, epochs=30, imgsz=640, batch=16,
    device=0 if torch.cuda.is_available() else 'cpu',
    patience=10, project='runs', name='cardd_detector', exist_ok=True,
)
m = model.val(data=DATA_YAML, split='test', verbose=False)
print(f"Retrained — mAP@50={m.box.map50:.3f}  mAP@50-95={m.box.map:.3f}")
import shutil
shutil.copy('runs/cardd_detector/weights/best.pt', 'best_model.pt')
detector = YOLO('best_model.pt')
print("Saved new best_model.pt")


# === SNIPPET 6 — Define the severity labeling rule ===========================
import cv2, numpy as np
from pathlib import Path

DAMAGE_NAMES = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat']
SEVERE_TYPES = {'glass shatter', 'lamp broken', 'tire flat', 'crack'}

def rule_severity(damage, box_frac):
    # box_frac = box area / image area (0..1)
    if damage in SEVERE_TYPES:
        return 'severe' if box_frac > 0.05 else 'moderate'
    if damage == 'scratch':
        return 'moderate' if box_frac > 0.10 else 'minor'
    if damage == 'dent':
        if box_frac > 0.12: return 'severe'
        if box_frac > 0.04: return 'moderate'
        return 'minor'
    return 'moderate'


# === SNIPPET 7 — Sanity-check the rule vs your hand-labels ===================
from PIL import Image
hand = {'minor': [], 'moderate': [], 'severe': []}
for c in hand:
    d = Path(c)
    if d.is_dir():
        for f in d.iterdir():
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                try:
                    w, h = Image.open(f).size
                    hand[c].append(w * h)
                except Exception:
                    pass
for c in hand:
    a = sorted(hand[c])
    if a:
        print(f"{c:9s} n={len(a):3d}  median area={a[len(a)//2]:,}")


# === SNIPPET 8 — Auto-label extra severity crops (weak supervision) ==========
OUT = Path('severity_auto')
for c in ['minor', 'moderate', 'severe']:
    (OUT / c).mkdir(parents=True, exist_ok=True)
counts = {'minor': 0, 'moderate': 0, 'severe': 0}
CONF = 0.30
SRC_DIRS = ['train/images', 'val/images']   # not test — keep test clean for the demo

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
import tensorflow as tf
import shutil, random
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG = (224, 224)
BATCH = 16
CLASSES = ['minor', 'moderate', 'severe']

train_root = Path('sev_train')
val_root = Path('sev_val')
for root in (train_root, val_root):
    if root.exists():
        shutil.rmtree(root)
    for c in CLASSES:
        (root / c).mkdir(parents=True, exist_ok=True)

random.seed(42)
for c in CLASSES:
    files = [f for f in Path(c).iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')] if Path(c).is_dir() else []
    random.shuffle(files)
    k = max(1, int(len(files) * 0.2))
    for f in files[:k]:
        shutil.copy(f, val_root / c / f.name)             # 20% hand -> trusted val
    for f in files[k:]:
        shutil.copy(f, train_root / c / ('hand_' + f.name))  # 80% hand -> train
for c in CLASSES:
    ad = Path('severity_auto') / c
    if ad.is_dir():
        for f in ad.iterdir():
            shutil.copy(f, train_root / c / f.name)        # auto -> train only
for c in CLASSES:
    print(f"{c:9s} train={len(list((train_root/c).iterdir())):4d}  val={len(list((val_root/c).iterdir())):3d}")


# === SNIPPET 10 — Train the severity classifier (MobileNetV2) ================
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
model.save('severity_model.h5')
open('severity_classes.txt', 'w').write("\n".join(CLASSES))
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
SEV = open('severity_classes.txt').read().split()

def assess(image_path, out='demo_output.jpg'):
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
    print(f"{len(found)} regions ->", found)
    return out

for i, t in enumerate(sorted(glob.glob('test/images/*.jpg'))[:3]):
    o = assess(t, f'demo_{i}.jpg')
    display(IPyImage(o))
