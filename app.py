import json
import os
import tempfile

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf
from matplotlib import pyplot as plt

from demo_config import CLASS_NAMES, DEMO_SAMPLES


SEED = 42
TARGET_SR = 16000
DURATION = 5
SAMPLES = TARGET_SR * DURATION
N_MELS = 128

APP_DIR = os.path.dirname(__file__)
DEMO_AUDIO_DIR = os.path.join(APP_DIR, "demo_audio")
DEMO_IMAGE_DIR = os.path.join(APP_DIR, "demo_images")
MODELS_DIR = os.path.join(APP_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "final_demo_model_best.keras")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "final_demo_model.keras")
RESULT_PATH = os.path.join(APP_DIR, "cnn_results_final_79.json")

np.random.seed(SEED)
tf.random.set_seed(SEED)


@st.cache_resource
def load_model():
    model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else DEFAULT_MODEL_PATH
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_results():
    if not os.path.exists(RESULT_PATH):
        fallback = os.path.join(APP_DIR, "cnn_results.json")
        if not os.path.exists(fallback):
            return None
        with open(fallback, "r", encoding="utf-8") as file:
            return json.load(file)

    with open(RESULT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_audio(path):
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
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
        n_mels=N_MELS,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
    return logmel.astype(np.float32)


def predict_audio(path, model):
    features = extract_logmel(path)
    features = np.expand_dims(features, axis=(0, -1))

    probs = model.predict(features, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]

    top_indices = np.argsort(probs)[::-1][:5]
    top_predictions = [
        (CLASS_NAMES[int(idx)], float(probs[idx]) * 100.0)
        for idx in top_indices
    ]
    return pred_label, float(probs[pred_idx]) * 100.0, top_predictions


def filter_strong_demo_samples(samples, model, min_confidence=80.0):
    filtered = []
    for sample in samples:
        audio_path = os.path.join(DEMO_AUDIO_DIR, sample["filename"])
        pred_label, confidence, _ = predict_audio(audio_path, model)
        if pred_label == sample["category"] and confidence >= min_confidence:
            item = dict(sample)
            item["demo_confidence"] = confidence
            filtered.append(item)
    return filtered


def prettify_label(label):
    return label.replace("_", " ").title()


def find_category_image(label):
    for extension in (".jpg", ".jpeg", ".png", ".webp"):
        image_path = os.path.join(DEMO_IMAGE_DIR, f"{label}{extension}")
        if os.path.exists(image_path):
            return image_path
    return None


def build_spectrogram_figure(path):
    spectrogram = extract_logmel(path)
    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    image = ax.imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Provided Audio Spectrogram")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel Bands")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def display_category_image(label, caption=""):
    image_path = find_category_image(label)
    if image_path:
        st.image(image_path, caption=caption or prettify_label(label), use_container_width=True)
    else:
        st.info(prettify_label(label))


def render_prediction_output(audio_path, source_label, model, caption="", source_category=None):
    pred_label, confidence, top_predictions = predict_audio(audio_path, model)

    st.success("Prediction complete")
    st.subheader("Three-Part Output")
    left_col, middle_col, right_col = st.columns(3)

    with left_col:
        st.markdown("**1. Provided Sample Image**")
        if source_category:
            display_category_image(source_category, caption=source_label)
        else:
            figure = build_spectrogram_figure(audio_path)
            st.pyplot(figure, clear_figure=True)
            st.caption(source_label)

    with middle_col:
        st.markdown("**2. Sound Names**")
        if source_category:
            st.metric("Provided Sample Sound", prettify_label(source_category))
        st.metric("Predicted Sound Name", prettify_label(pred_label))
        st.metric("Prediction Confidence", f"{confidence:.2f}%")
        if caption:
            st.caption(caption)

    with right_col:
        st.markdown("**3. Predicted Image of Audio**")
        display_category_image(pred_label, caption=prettify_label(pred_label))

    st.subheader("Top 5 Softmax Output")
    prediction_columns = st.columns(5)
    for column, (label, score) in zip(prediction_columns, top_predictions):
        with column:
            st.markdown(
                f"""
                <div style="border:1px solid #ddd;border-radius:16px;padding:18px;text-align:center;background:#fafafa;min-height:150px;">
                    <div style="margin-top:10px;font-weight:600;">{prettify_label(label)}</div>
                    <div style="margin-top:10px;color:#444;">{score:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.set_page_config(page_title="Environmental Audio Recognition", page_icon="🎧", layout="wide")
st.title("Environmental Audio Recognition for Surveillance")
st.write("Upload a `.wav` file or use the prepared demo samples to classify environmental sounds.")

if not os.path.exists(DEFAULT_MODEL_PATH) and not os.path.exists(BEST_MODEL_PATH):
    st.error("No model file found in the models folder.")
    st.stop()

model = load_model()
results = load_results()
demo_samples = filter_strong_demo_samples(DEMO_SAMPLES, model, min_confidence=80.0)

with st.expander("Project Summary", expanded=False):
    st.write("Dataset: ESC-50")
    st.write("Classes: 50")
    st.write("Approach: Log-mel spectrogram + CNN")
    if results is not None:
        st.write(f"Mean Accuracy: {results['mean_accuracy']:.4f}")
        st.write(f"Mean Precision: {results['mean_precision_macro']:.4f}")
        st.write(f"Mean Recall: {results['mean_recall_macro']:.4f}")
        st.write(f"Mean F1-score: {results['mean_f1_macro']:.4f}")
        st.write(f"Mean ROC-AUC: {results['mean_roc_auc_ovr']:.4f}")

demo_tab, upload_tab = st.tabs(["Demo Sample", "Upload Your Own Audio"])

with demo_tab:
    st.markdown("### Prepared Demo Samples")
    if not demo_samples:
        st.warning("No prepared sample currently meets the 80% confidence rule with the included model.")
    else:
        st.caption("Only prepared samples with at least 80% correct confidence are shown.")
        for start in range(0, len(demo_samples), 5):
            row_samples = demo_samples[start : start + 5]
            row_columns = st.columns(5)
            for column, sample in zip(row_columns, row_samples):
                with column:
                    display_category_image(sample["category"], caption=prettify_label(sample["category"]))

        selected_title = st.selectbox(
            "Choose a prepared demo sample",
            options=[sample["title"] for sample in demo_samples],
        )
        selected_sample = next(sample for sample in demo_samples if sample["title"] == selected_title)
        selected_audio = os.path.join(DEMO_AUDIO_DIR, selected_sample["filename"])

        st.audio(selected_audio, format="audio/wav")
        st.caption(f"Selected file: {selected_sample['filename']}")
        st.caption(f"Prepared demo confidence: {selected_sample['demo_confidence']:.2f}%")

        if st.button("Run Demo Prediction", type="primary"):
            with st.spinner("Analyzing demo audio..."):
                render_prediction_output(
                    selected_audio,
                    selected_sample["filename"],
                    model,
                    caption=selected_sample["caption"],
                    source_category=selected_sample["category"],
                )

with upload_tab:
    st.markdown("### Custom Audio Upload")
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        try:
            if st.button("Predict Uploaded Audio", type="primary"):
                with st.spinner("Analyzing uploaded audio..."):
                    render_prediction_output(temp_path, uploaded_file.name, model)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
