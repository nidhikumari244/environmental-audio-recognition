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

CLASS_DESCRIPTIONS = {
    "dog": "Animal sound that can support presence awareness in outdoor or residential monitoring.",
    "rain": "Ambient weather sound usually interpreted as non-critical background environmental activity.",
    "helicopter": "Airborne mechanical sound useful for traffic, public safety, or emergency awareness.",
    "siren": "Emergency-response sound with strong surveillance relevance and immediate alert value.",
    "footsteps": "Human movement cue that may indicate presence, motion, or access in a monitored area.",
    "hand_saw": "Tool-use sound that can indicate maintenance, construction, or suspicious manual activity.",
    "clock_alarm": "Structured alert-like indoor sound with moderate relevance for smart monitoring.",
    "glass_breaking": "Abnormal impact sound that may indicate intrusion, damage, or a security breach.",
    "train": "Infrastructure or transit sound that provides environmental context in transport zones.",
    "crackling_fire": "Hazard-oriented sound associated with burning material and possible fire incidents.",
    "crying_baby": "Distress-related human sound relevant in residential or care-monitoring environments.",
    "cat": "Low-priority animal sound that still provides environmental context.",
    "car_horn": "Urban alert sound relevant to traffic monitoring and outdoor disturbance detection.",
    "thunderstorm": "Natural weather disturbance that explains background conditions rather than intrusion.",
    "church_bells": "Public-space ambience sound commonly associated with community or religious environments.",
}

SURVEILLANCE_RELEVANCE = {
    "glass_breaking": ("High", "Possible intrusion or physical damage event."),
    "siren": ("High", "Emergency-related sound with direct safety importance."),
    "footsteps": ("High", "Can indicate nearby motion or human presence."),
    "crackling_fire": ("High", "Potential hazard indicator requiring attention."),
    "car_horn": ("Medium", "Useful for road activity and disturbance monitoring."),
    "crying_baby": ("Medium", "May indicate distress in indoor or care settings."),
    "hand_saw": ("Medium", "Could indicate active tool use or suspicious work."),
    "helicopter": ("Medium", "Relevant in public safety or transport scenarios."),
    "train": ("Low", "Mostly contextual transport sound."),
    "dog": ("Low", "Environmental context cue with occasional presence relevance."),
    "rain": ("Low", "Ambient weather condition with low direct security relevance."),
    "clock_alarm": ("Low", "Contextual indoor alert sound."),
    "cat": ("Low", "Background animal ambience."),
    "thunderstorm": ("Low", "Natural weather event rather than human activity."),
    "church_bells": ("Low", "Routine ambient public-space sound."),
}


APP_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(56, 189, 248, 0.16), transparent 28%),
            radial-gradient(circle at top left, rgba(34, 197, 94, 0.10), transparent 24%),
            linear-gradient(180deg, #0b1220 0%, #111827 45%, #0f172a 100%);
    }

    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 3rem;
        max-width: 1220px;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(14, 116, 144, 0.26), rgba(22, 163, 74, 0.14));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 26px;
        padding: 1.6rem 1.6rem 1.25rem;
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.22);
        margin-bottom: 1.2rem;
    }

    .hero-kicker {
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-size: 0.74rem;
        color: #7dd3fc;
        margin-bottom: 0.55rem;
        font-weight: 800;
    }

    .hero-title {
        font-size: 2.75rem;
        line-height: 1.06;
        font-weight: 900;
        color: #f8fafc;
        margin: 0;
    }

    .hero-subtitle {
        margin-top: 0.85rem;
        font-size: 1rem;
        line-height: 1.7;
        color: #dbeafe;
        max-width: 56rem;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 1.1rem;
    }

    .stat-card {
        background: rgba(15, 23, 42, 0.56);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 18px;
        padding: 0.9rem 1rem;
    }

    .stat-label {
        color: #93c5fd;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
    }

    .stat-value {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 900;
        margin-top: 0.38rem;
    }

    .section-shell {
        background: rgba(15, 23, 42, 0.52);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 22px;
        padding: 1rem 1.15rem 1.15rem;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.18);
        margin: 1rem 0 1.1rem;
    }

    .section-kicker {
        color: #7dd3fc;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.45rem;
        line-height: 1.2;
        font-weight: 850;
        margin-bottom: 0.35rem;
    }

    .section-copy {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.58;
    }

    .minor-note {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin-top: -0.1rem;
        margin-bottom: 0.8rem;
    }

    .insight-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 1rem 0 1.1rem;
    }

    .insight-card {
        background: rgba(15, 23, 42, 0.58);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.16);
    }

    .insight-label {
        color: #93c5fd;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .insight-value {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 850;
        line-height: 1.25;
        margin-bottom: 0.35rem;
    }

    .insight-copy {
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        border-radius: 999px;
        padding: 0.5rem 0.8rem;
        font-size: 0.84rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    .status-chip.high {
        background: rgba(34, 197, 94, 0.16);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.26);
    }

    .status-chip.medium {
        background: rgba(250, 204, 21, 0.16);
        color: #fde68a;
        border: 1px solid rgba(250, 204, 21, 0.26);
    }

    .status-chip.low {
        background: rgba(248, 113, 113, 0.16);
        color: #fca5a5;
        border: 1px solid rgba(248, 113, 113, 0.26);
    }

    .softmax-card {
        border-radius: 22px;
        padding: 1rem;
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(226, 232, 240, 0.98));
        box-shadow: 0 18px 36px rgba(15, 23, 42, 0.20);
        border: 1px solid rgba(148, 163, 184, 0.22);
    }

    .softmax-card.top {
        background: linear-gradient(160deg, #ecfeff 0%, #cffafe 42%, #e0f2fe 100%);
        border: 2px solid rgba(8, 145, 178, 0.5);
        transform: translateY(-4px);
    }

    .softmax-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2rem;
        height: 2rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.08);
        color: #0f172a;
        font-size: 0.82rem;
        font-weight: 850;
        margin-bottom: 0.75rem;
    }

    .softmax-card.top .softmax-rank {
        background: rgba(14, 165, 233, 0.18);
        color: #0c4a6e;
    }

    .softmax-label {
        color: #0f172a;
        font-size: 1.05rem;
        line-height: 1.25;
        font-weight: 900;
        margin-bottom: 0.6rem;
    }

    .softmax-score {
        color: #0f172a;
        font-size: 1.55rem;
        font-weight: 950;
        margin-bottom: 0.75rem;
    }

    .softmax-bar-shell {
        width: 100%;
        height: 0.72rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.08);
        overflow: hidden;
        margin-bottom: 0.55rem;
    }

    .softmax-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #06b6d4, #0ea5e9);
    }

    .softmax-card:not(.top) .softmax-bar-fill {
        background: linear-gradient(90deg, #94a3b8, #64748b);
    }

    .softmax-footnote {
        color: #475569;
        font-size: 0.83rem;
        line-height: 1.45;
    }

    .softmax-wrap {
        margin-top: 0.8rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 999px 999px 0 0;
        padding: 0.85rem 1.1rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(8, 145, 178, 0.18);
        border-color: rgba(56, 189, 248, 0.28);
    }

    .footer-card {
        margin-top: 1.8rem;
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.42);
        border: 1px solid rgba(148, 163, 184, 0.12);
        color: #94a3b8;
        font-size: 0.9rem;
        text-align: center;
    }

    @media (max-width: 1100px) {
        .stat-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .insight-grid {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 720px) {
        .hero-title {
            font-size: 2rem;
        }

        .stat-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
"""


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


def get_confidence_status(confidence):
    if confidence >= 80:
        return "High Confidence", "high"
    if confidence >= 55:
        return "Moderate Confidence", "medium"
    return "Low Confidence", "low"


def get_class_description(label):
    return CLASS_DESCRIPTIONS.get(
        label,
        "Environmental sound class detected by the CNN model using Log-Mel spectrogram features.",
    )


def get_surveillance_relevance(label):
    return SURVEILLANCE_RELEVANCE.get(
        label,
        ("Medium", "Useful contextual sound for environmental monitoring."),
    )


def render_summary_cards(results):
    if results is None:
        return

    stats = [
        ("Mean Accuracy", f"{results['mean_accuracy'] * 100:.2f}%"),
        ("Mean Precision", f"{results['mean_precision_macro'] * 100:.2f}%"),
        ("Mean Recall", f"{results['mean_recall_macro'] * 100:.2f}%"),
        ("Mean F1-Score", f"{results['mean_f1_macro'] * 100:.2f}%"),
    ]

    stats_markup = "".join(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>
        """
        for label, value in stats
    )

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">Deep Audio Intelligence</div>
            <h1 class="hero-title">Environmental Audio Recognition for Surveillance</h1>
            <div class="hero-subtitle">
                A polished environmental sound classification workspace for live demo presentation, softmax inspection,
                and surveillance-focused audio analysis using CNN-based inference.
            </div>
            <div class="stat-grid">
                {stats_markup}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(kicker, title, copy):
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <div class="section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_softmax_cards(top_predictions):
    render_section_header(
        "Softmax Confidence Distribution",
        "Top 5 Model Outputs",
        "The softmax layer converts the CNN output into class probabilities. The highest score becomes the final prediction, while the remaining cards show the nearest alternatives.",
    )
    st.markdown('<div class="softmax-wrap"></div>', unsafe_allow_html=True)
    softmax_columns = st.columns(5)
    for column, (rank, (label, score)) in zip(softmax_columns, enumerate(top_predictions, start=1)):
        width = max(score, 3.0)
        card_class = "softmax-card top" if rank == 1 else "softmax-card"
        footnote = "Highest-confidence class selected by the model." if rank == 1 else "Alternative competing class."
        with column:
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div>
                        <div class="softmax-rank">#{rank}</div>
                        <div class="softmax-label">{prettify_label(label)}</div>
                        <div class="softmax-score">{score:.2f}%</div>
                    </div>
                    <div>
                        <div class="softmax-bar-shell">
                            <div class="softmax-bar-fill" style="width:{width:.2f}%"></div>
                        </div>
                        <div class="softmax-footnote">{footnote}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_prediction_insights(pred_label, confidence):
    confidence_text, confidence_class = get_confidence_status(confidence)
    relevance_level, relevance_note = get_surveillance_relevance(pred_label)
    class_description = get_class_description(pred_label)

    st.markdown(
        f"""
        <div class="insight-grid">
            <div class="insight-card">
                <div class="insight-label">Confidence Status</div>
                <div class="insight-value">{confidence_text}</div>
                <div class="status-chip {confidence_class}">{confidence:.2f}% model confidence</div>
            </div>
            <div class="insight-card">
                <div class="insight-label">Surveillance Relevance</div>
                <div class="insight-value">{relevance_level}</div>
                <div class="insight-copy">{relevance_note}</div>
            </div>
            <div class="insight-card">
                <div class="insight-label">Class Interpretation</div>
                <div class="insight-value">{prettify_label(pred_label)}</div>
                <div class="insight-copy">{class_description}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_output(audio_path, source_label, model, caption="", source_category=None):
    pred_label, confidence, top_predictions = predict_audio(audio_path, model)

    st.success("Prediction complete")
    render_section_header(
        "Prediction Workspace",
        "Three-Part Output",
        "Inspect the input representation, the model decision, and the visual class output in one place for a cleaner and more explainable demo flow.",
    )

    left_col, middle_col, right_col = st.columns([1.15, 0.8, 1.15])

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

    render_prediction_insights(pred_label, confidence)
    render_softmax_cards(top_predictions)


st.set_page_config(page_title="Environmental Audio Recognition", page_icon="EA", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

if not os.path.exists(DEFAULT_MODEL_PATH) and not os.path.exists(BEST_MODEL_PATH):
    st.error("No model file found in the models folder.")
    st.stop()

model = load_model()
results = load_results()
demo_samples = filter_strong_demo_samples(DEMO_SAMPLES, model, min_confidence=80.0)

render_summary_cards(results)
st.caption(
    "Upload a `.wav` file or choose a prepared sample to inspect the predicted class, confidence, and softmax competition."
)

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
    st.markdown(
        '<div class="minor-note">Curated, high-confidence demo clips that make the live presentation more stable and easier to explain.</div>',
        unsafe_allow_html=True,
    )

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
    st.markdown(
        '<div class="minor-note">Upload your own `.wav` file to inspect the spectrogram, predicted label, confidence, and softmax alternatives.</div>',
        unsafe_allow_html=True,
    )
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

st.markdown(
    """
    <div class="footer-card">
        Built with Python, TensorFlow, Librosa, Streamlit, and the ESC-50 dataset.
        This interface combines CNN inference, softmax inspection, and surveillance-focused interpretation for live demonstration.
    </div>
    """,
    unsafe_allow_html=True,
)
