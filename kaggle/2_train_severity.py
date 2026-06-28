"""
Stage 2 — Train the damage-severity classifier (minor / moderate / severe).

Input: cropped damage regions sorted into class folders:
    severity_data/minor/*.jpg
    severity_data/moderate/*.jpg
    severity_data/severe/*.jpg

This CNN is tiny and trains fine on CPU (~minutes). It uses a MobileNetV2
backbone (transfer learning) so it generalizes from the small ~195-image set
far better than a from-scratch CNN would.

Output: severity_model.h5
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

DATA_DIR = "severity_data"   # contains minor/ moderate/ severe/
IMG_SIZE = (224, 224)
BATCH = 16
EPOCHS = 25
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH,
)
class_names = train_ds.class_names
print("Severity classes (index order matters for inference):", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(200).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# Light augmentation — the dataset is small, so this matters.
augment = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False  # freeze backbone; train only the head on this small set

model = models.Sequential([
    layers.Rescaling(1.0 / 127.5, offset=-1),   # MobileNetV2 expects [-1, 1]
    augment,
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

val_loss, val_acc = model.evaluate(val_ds)
print(f"\nFinal validation accuracy: {val_acc:.3f}")

model.save("severity_model.h5")
# Persist class order so inference labels never get mismatched.
with open("severity_classes.txt", "w") as f:
    f.write("\n".join(class_names))
print("Saved severity_model.h5 and severity_classes.txt")
