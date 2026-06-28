"""
================================================================================
 Vehicle Damage Detection & Severity — COLAB (data pulled from Kaggle)
================================================================================
Each block below is ONE Colab cell. Run them top to bottom.

This version runs in Google Colab but downloads the 2.9 GB CarDD dataset
directly from Kaggle via the API (no slow Drive upload). You only upload the
small 21.6 MB bundle (best_model.pt + your hand-labeled crops).

ONE-TIME PREP — get a Kaggle API token:
  1. kaggle.com -> avatar -> Settings -> API -> "Create New Token".
  2. Copy the KGAT_... value. Snippet 1 will prompt you to paste it (hidden).

For Snippet 2: open the Files panel (folder icon, left sidebar) and DRAG
kaggle/upload_to_kaggle.zip into it. (Do NOT use the files.upload() widget —
it crashes the browser on files this size.)

This trains the severity model on your 195 hand-labeled crops only — no
auto-labeling step, no .cache() — so it won't run out of RAM.

Set Colab Runtime > Change runtime type > T4 GPU before running.
================================================================================
"""


# === SNIPPET 1 — GPU check + install + Kaggle token =========================
import torch
print("CUDA available:", torch.cuda.is_available(), "|",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

get_ipython().system('pip -q install ultralytics==8.* kaggle >/dev/null')

# Kaggle's new tokens are a KGAT_... string (Settings > API > Create New Token).
# getpass hides it as you type — paste it at the prompt, do NOT hardcode it here.
import os, getpass
os.environ['KAGGLE_API_TOKEN'] = getpass.getpass("Paste your Kaggle token (KGAT_...): ").strip()
print("Kaggle token set.")


# === SNIPPET 2 — Unpack your bundle (drag the zip into the Files panel) ======
# DO NOT use files.upload() — its browser widget crashes on >~10MB (OOM).
# Instead: open the Files panel (folder icon, left sidebar) and DRAG
# upload_to_kaggle.zip into it (drops at /content/). Then run this cell.
import zipfile
ZIP = '/content/upload_to_kaggle.zip'
assert os.path.exists(ZIP), "Drag upload_to_kaggle.zip into the Files panel first, then re-run."
os.makedirs('/content/mydata', exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall('/content/mydata')
print("Extracted to /content/mydata:", sorted(os.listdir('/content/mydata')))
for c in ['minor', 'moderate', 'severe']:
    p = f'/content/mydata/{c}'
    print(f"  {c}: {len(os.listdir(p)) if os.path.isdir(p) else 'MISSING'}")


# === SNIPPET 3 — Download the CarDD dataset from Kaggle =====================
# If this slug 404s, run:  !kaggle datasets list -s "cardd yolo"
# and replace DATASET_SLUG with the owner/name it prints.
DATASET_SLUG = 'gabrielfcarvalho/cardd-with-yolo-annotations-images-labels'
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


# === SNIPPET 7 — Train the severity classifier (MobileNetV2, hand-labels) ====
# Trains on your 195 hand-labeled crops only (minor/moderate/severe).
# NO auto-labeling and NO .cache() -> avoids the out-of-memory crash.
# MobileNetV2 transfer learning + augmentation handles this small set well.
import cv2, numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

DAMAGE_NAMES = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat']
CLASSES = ['minor', 'moderate', 'severe']
IMG, BATCH = (224, 224), 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    HAND, validation_split=0.2, subset='training', seed=42,
    image_size=IMG, batch_size=BATCH, class_names=CLASSES, label_mode='int')
val_ds = tf.keras.utils.image_dataset_from_directory(
    HAND, validation_split=0.2, subset='validation', seed=42,
    image_size=IMG, batch_size=BATCH, class_names=CLASSES, label_mode='int', shuffle=False)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)   # no .cache() -> low RAM
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

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


# === SNIPPET 8 — Confusion matrix on the trusted hand-labeled val set ========
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


# === SNIPPET 9 — End-to-end demo =============================================
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
