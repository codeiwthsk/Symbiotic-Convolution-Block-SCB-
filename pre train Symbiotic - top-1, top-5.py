# ==========================================================
# MobileNet + Symbiotic Convolution Block on CIFAR-100
# With Training Curves, Top-1 / Top-5 Accuracy & Error
# ==========================================================

import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, TopKCategoricalAccuracy
from tensorflow.keras.datasets import cifar100

# ==========================================================
# CUSTOM SYMBIOTIC CONVOLUTION BLOCK
# ==========================================================
def symbio_swish(x, beta=1.5):
    return x * tf.sigmoid(beta * x)

def SymbioticConv(x, filters, kernel_size=3, strides=1):
    shared = layers.Conv2D(filters // 2, kernel_size, strides=strides,
                           padding='same', use_bias=False,
                           kernel_initializer='he_normal')(x)
    mutated = layers.Conv2D(filters - filters // 2, kernel_size, strides=strides,
                            padding='same', use_bias=False,
                            kernel_initializer='random_normal')(x)
    x = layers.Add()([shared, mutated])
    x = layers.BatchNormalization()(x)
    x = layers.Activation(symbio_swish)(x)
    return x

# ==========================================================
# CIFAR-100 DATASET
# ==========================================================
NUM_CLASSES = 100
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
INPUT_SHAPE = (224, 224, 3)

class_names = [f"class_{i}" for i in range(NUM_CLASSES)]

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")

y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test  = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(50000)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = val_ds

# ==========================================================
# CALLBACKS
# ==========================================================
callbacks = [
    ModelCheckpoint("best_mobilenet_symbiotic_cifar100.h5",
                    monitor="val_loss", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=7,
                  restore_best_weights=True, verbose=1),
    CSVLogger("training_log_cifar100.csv")
]

# ==========================================================
# MODEL
# ==========================================================
base_model = applications.MobileNet(
    weights="imagenet",
    include_top=False,
    input_shape=INPUT_SHAPE
)
base_model.trainable = True

x = base_model.output
x = SymbioticConv(x, filters=512)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(base_model.input, outputs)

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        Precision(name="precision"),
        Recall(name="recall"),
        TopKCategoricalAccuracy(k=5, name="top5_accuracy")
    ]
)

model.summary()

# ==========================================================
# TRAIN
# ==========================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ==========================================================
# MULTI-METRIC PLOTS (AUTO-SAVE)
# ==========================================================
def plot_all_metrics(history):
    epochs = range(1, len(history.history["loss"]) + 1)

    train_f1 = np.array(history.history["precision"]) * np.array(history.history["recall"]) * 2 / \
               (np.array(history.history["precision"]) + np.array(history.history["recall"]) + 1e-8)

    val_f1 = np.array(history.history["val_precision"]) * np.array(history.history["val_recall"]) * 2 / \
             (np.array(history.history["val_precision"]) + np.array(history.history["val_recall"]) + 1e-8)

    plt.figure(figsize=(18, 10))

    # Top-1 Accuracy
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history.history["accuracy"], label="Train")
    plt.plot(epochs, history.history["val_accuracy"], label="Val")
    plt.title("Top-1 Accuracy")
    plt.legend()

    # Top-5 Accuracy
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history.history["top5_accuracy"], label="Train")
    plt.plot(epochs, history.history["val_top5_accuracy"], label="Val")
    plt.title("Top-5 Accuracy")
    plt.legend()

    # Loss
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history.history["loss"], label="Train")
    plt.plot(epochs, history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    # Precision
    plt.subplot(2, 3, 4)
    plt.plot(epochs, history.history["precision"], label="Train")
    plt.plot(epochs, history.history["val_precision"], label="Val")
    plt.title("Precision")
    plt.legend()

    # Recall
    plt.subplot(2, 3, 5)
    plt.plot(epochs, history.history["recall"], label="Train")
    plt.plot(epochs, history.history["val_recall"], label="Val")
    plt.title("Recall")
    plt.legend()

    # F1-score
    plt.subplot(2, 3, 6)
    plt.plot(epochs, train_f1, label="Train")
    plt.plot(epochs, val_f1, label="Val")
    plt.title("F1-Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig("cifar100_training_metrics.png", dpi=300)
    plt.show()

plot_all_metrics(history)

# ==========================================================
# EVALUATION (TOP-1 / TOP-5 ERROR)
# ==========================================================
test_loss, test_top1, test_prec, test_rec, test_top5 = model.evaluate(test_ds)

print("\n================ TEST RESULTS ================")
print(f"Top-1 Accuracy : {test_top1*100:.2f}%")
print(f"Top-1 Error    : {(1-test_top1)*100:.2f}%")
print(f"Top-5 Accuracy : {test_top5*100:.2f}%")
print(f"Top-5 Error    : {(1-test_top5)*100:.2f}%")
print("=============================================")

# ==========================================================
# REPORT, CONFUSION MATRIX, KAPPA
# ==========================================================
y_pred = np.argmax(model.predict(test_ds), axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix – CIFAR-100")
plt.show()

print(f"Cohen’s Kappa: {cohen_kappa_score(y_true, y_pred):.4f}")

# ==========================================================
# INFERENCE SPEED
# ==========================================================
dummy = np.random.rand(1, 224, 224, 3).astype("float32")

start = time.time()
for _ in range(100):
    _ = model.predict(dummy, verbose=0)
elapsed = (time.time() - start) / 100

print(f"Inference Time: {elapsed*1000:.2f} ms | FPS: {1/elapsed:.1f}")
print(f"Total Parameters: {model.count_params():,}")
