import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import json

# Paths
DATASET_PATH = "Dataset"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "bird_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

# Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Training data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=40,
    zoom_range=0.3,
    horizontal_flip=True,
    shear_range=0.2,
    brightness_range=[0.6, 1.4],
    width_shift_range=0.15,
    height_shift_range=0.15
)

# Validation data: only rescaling
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("Class indices:", train_data.class_indices)

# Load pretrained base model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = True

# Freeze most layers, fine-tune only the last part
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation="softmax")
])

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
os.makedirs(MODELS_DIR, exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save final model again to ensure file exists
model.save(MODEL_PATH)

# Save class names in correct order
class_names = list(train_data.class_indices.keys())
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=2)

print("Saved class names:", class_names)
print("Model and class names saved successfully!")