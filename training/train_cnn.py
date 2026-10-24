"""
training/train_cnn.py
─────────────────────
Optional custom CNN training on a small dataset (e.g., cats vs dogs subset).

Usage
-----
1. Download a dataset (see README) and place it under:
       training/data/
           train/
               cat/  ...images...
               dog/  ...images...
           val/
               cat/  ...images...
               dog/  ...images...
           test/
               cat/  ...images...
               dog/  ...images...

2. From the project root:
       python training/train_cnn.py

3. After training, results are saved to:
       models/custom_cnn/          ← SavedModel
       models/eval_results.json    ← metrics loaded by the main app
       training/history.json       ← accuracy/loss curves
       training/plots/             ← PNG plots

IMPORTANT
---------
This custom model recognises only the classes you train it on (e.g. cat, dog).
The pretrained MobileNetV2 in app.py recognises 1 000 ImageNet classes.
They are separate models with different label spaces.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path so we can resolve relative directories
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR    = Path(__file__).parent / "data"
MODELS_DIR  = ROOT / "models"
PLOTS_DIR   = Path(__file__).parent / "plots"
HISTORY_JSON = Path(__file__).parent / "history.json"
EVAL_JSON   = MODELS_DIR / "eval_results.json"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-3


def check_data():
    for split in ("train", "val", "test"):
        path = DATA_DIR / split
        if not path.exists():
            print(f"[ERROR] Missing directory: {path}")
            print("       Please prepare your dataset first — see README.")
            sys.exit(1)
    print("[OK]  Dataset directories found.")


def build_datasets():
    import tensorflow as tf

    rescale = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR / "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    ).map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR / "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    ).map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR / "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    ).map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    class_names = train_ds.class_names
    print(f"[OK]  Classes: {class_names}")
    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes: int):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="CustomCNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def train(model, train_ds, val_ds):
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "custom_cnn_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    return history


def save_history(history):
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(HISTORY_JSON, "w") as f:
        json.dump(hist_dict, f, indent=2)
    print(f"[OK]  Training history saved → {HISTORY_JSON}")


def plot_history(history):
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Custom CNN – Training History", fontsize=14)

        # Accuracy
        axes[0].plot(history.history["accuracy"], label="Train")
        axes[0].plot(history.history["val_accuracy"], label="Val")
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(history.history["loss"], label="Train")
        axes[1].plot(history.history["val_loss"], label="Val")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out = PLOTS_DIR / "training_history.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        print(f"[OK]  Plot saved → {out}")
    except ImportError:
        print("[WARN] matplotlib not installed; skipping plots.")


def evaluate(model, test_ds, class_names):
    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"\n[RESULT] Test Loss:     {loss:.4f}")
    print(f"[RESULT] Test Accuracy: {acc * 100:.2f}%")

    # Per-class metrics via sklearn if available
    per_class = {}
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        y_true, y_pred = [], []
        for x_batch, y_batch in test_ds:
            preds = model.predict(x_batch, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1).tolist())
            y_true.extend(np.argmax(y_batch.numpy(), axis=1).tolist())

        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        cm     = confusion_matrix(y_true, y_pred).tolist()
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        per_class = report
    except ImportError:
        print("[WARN] scikit-learn not installed; skipping detailed metrics.")
        cm = []

    results = {
        "model_name":    "CustomCNN",
        "model_type":    "Custom (trained on local dataset)",
        "dataset":       "Custom local dataset",
        "class_names":   class_names,
        "test_loss":     round(float(loss), 4),
        "test_accuracy": round(float(acc), 4),
        "per_class":     per_class,
        "confusion_matrix": cm,
        "note": (
            "These results were measured on a held-out test set by training/train_cnn.py. "
            "They reflect this custom model, not MobileNetV2."
        ),
    }

    with open(EVAL_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK]  Evaluation results saved → {EVAL_JSON}")
    return results


def save_model(model, class_names):
    model.save(str(MODELS_DIR / "custom_cnn"))
    with open(MODELS_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)
    print(f"[OK]  Model saved → {MODELS_DIR / 'custom_cnn'}")


def main():
    print("=" * 60)
    print("  CNN Vision Lab – Custom Training Script")
    print("=" * 60)
    check_data()
    train_ds, val_ds, test_ds, class_names = build_datasets()
    model = build_model(num_classes=len(class_names))
    history = train(model, train_ds, val_ds)
    save_history(history)
    plot_history(history)
    evaluate(model, test_ds, class_names)
    save_model(model, class_names)
    print("\n[DONE] Training complete.")


if __name__ == "__main__":
    main()
