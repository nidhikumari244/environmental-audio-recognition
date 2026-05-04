import json
import os

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from tensorflow.keras import layers, models, callbacks

SEED = 42
TARGET_SR = 16000
DURATION = 5
SAMPLES = TARGET_SR * DURATION
N_MELS = 128
NUM_FOLDS = 5

BASE_DIR = r"C:\Users\nidhi\OneDrive\Desktop\environmental_sound\ESC-50"
META_PATH = os.path.join(BASE_DIR, "meta", "esc50.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
RESULT_DIR = os.path.join(BASE_DIR, "results_cnn_esc50")
os.makedirs(RESULT_DIR, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

df = pd.read_csv(META_PATH)
df["audio_path"] = df["filename"].apply(lambda x: os.path.join(AUDIO_DIR, x))

le = LabelEncoder()
df["label"] = le.fit_transform(df["category"])
num_classes = len(le.classes_)


def load_audio(path):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]
    return y


def extract_logmel(path):
    y = load_audio(path)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_fft=1024,
        hop_length=512,
        n_mels=N_MELS
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
    return logmel.astype(np.float32)


print("Extracting log-mel spectrograms...")
X = np.array([extract_logmel(path) for path in df["audio_path"]])
X = X[..., np.newaxis]
y = df["label"].values

print("Feature shape:", X.shape)


def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
fold_results = []

all_true = []
all_probs = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n========== FOLD {fold} ==========")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = build_model(X.shape[1:], num_classes)

    cb = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=cb,
        verbose=1
    )

    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(
            label_binarize(y_test, classes=np.arange(num_classes)),
            probs,
            multi_class="ovr"
        )
    except Exception:
        auc = 0.0

    print(f"Fold {fold} Accuracy: {acc:.4f}")
    print(f"Fold {fold} Precision: {prec:.4f}")
    print(f"Fold {fold} Recall: {rec:.4f}")
    print(f"Fold {fold} F1-score: {f1:.4f}")
    print(f"Fold {fold} ROC-AUC: {auc:.4f}")

    fold_results.append({
        "fold": fold,
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "roc_auc_ovr": float(auc)
    })

    all_true.extend(y_test)
    all_probs.extend(probs)

summary = {
    "mean_accuracy": float(np.mean([r["accuracy"] for r in fold_results])),
    "mean_precision_macro": float(np.mean([r["precision_macro"] for r in fold_results])),
    "mean_recall_macro": float(np.mean([r["recall_macro"] for r in fold_results])),
    "mean_f1_macro": float(np.mean([r["f1_macro"] for r in fold_results])),
    "mean_roc_auc_ovr": float(np.mean([r["roc_auc_ovr"] for r in fold_results])),
    "fold_results": fold_results
}

print("\n========== FINAL RESULTS ==========")
print(f"Mean Accuracy      : {summary['mean_accuracy']:.4f}")
print(f"Mean Precision     : {summary['mean_precision_macro']:.4f}")
print(f"Mean Recall        : {summary['mean_recall_macro']:.4f}")
print(f"Mean F1-score      : {summary['mean_f1_macro']:.4f}")
print(f"Mean ROC-AUC (OVR) : {summary['mean_roc_auc_ovr']:.4f}")

with open(os.path.join(RESULT_DIR, "cnn_results.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
