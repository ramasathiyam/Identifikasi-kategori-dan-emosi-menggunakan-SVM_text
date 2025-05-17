import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
import string

# Set page config
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .title {
        font-size: 2.5em;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .prediction-box {
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f0f2f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .positive {
        color: #2ecc71;
    }
    .negative {
        color: #e74c3c;
    }
    .neutral {
        color: #3498db;
    }
    .emotion-score {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load models and vectorizer
@st.cache_resource
def load_models():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    models = {}
    model_files = {
        'anger': 'saved_models/svm_model_label_anger.pkl',
        'fear': 'saved_models/svm_model_label_fear.pkl',
        'happy': 'saved_models/svm_model_label_happy.pkl',
        'love': 'saved_models/svm_model_label_love.pkl',
        'sadness': 'saved_models/svm_model_label_sadness.pkl'
    }

    for emotion, path in model_files.items():
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
                models[emotion] = {
                    'w': model_data['w'],
                    'b': model_data['b']
                }
        except FileNotFoundError:
            st.error(f"Model file not found: {path}")

    return vectorizer, models

# Text cleaning function
factory = StopWordRemoverFactory()
stopwords_id = set(factory.get_stop_words())

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if len(word) > 1])
    tokens = [word for word in text.split() if word not in stopwords_id]
    return ' '.join(tokens)

# Prediction function
def predict_emotion(text, vectorizer, models):
    cleaned_text = clean_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    scores = {}

    for emotion, model in models.items():
        score = text_vectorized.dot(model['w']) - model['b']
        scores[emotion] = score[0]

    predicted_emotion = max(scores, key=scores.get)
    return predicted_emotion, scores

# Main app
def main():
    st.markdown('<h1 class="title">Emotion Classifier for Indonesian Text</h1>', unsafe_allow_html=True)
    vectorizer, models = load_models()

    st.sidebar.title("About")
    st.sidebar.info("""
    This app classifies Indonesian text into one of these emotions:
    - 😠 Anger
    - 😨 Fear
    - 😊 Happy
    - 😍 Love
    - 😢 Sadness

    The model uses a custom SVM implementation trained on Indonesian customer reviews.
    """)

    st.sidebar.title("Settings")
    show_details = st.sidebar.checkbox("Show prediction details", value=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Enter your text")
        user_input = st.text_area("Type or paste Indonesian text here:", height=150)

        if st.button("Classify Emotion"):
            if user_input.strip() == "":
                st.warning("Please enter some text to analyze")
            else:
                with st.spinner("Analyzing..."):
                    emotion, scores = predict_emotion(user_input, vectorizer, models)

                    st.subheader("Prediction Result")
                    emotion_icons = {
                        'anger': '😠',
                        'fear': '😨',
                        'happy': '😊',
                        'love': '😍',
                        'sadness': '😢'
                    }
                    emotion_classes = {
                        'anger': 'negative',
                        'fear': 'negative',
                        'happy': 'positive',
                        'love': 'positive',
                        'sadness': 'negative'
                    }

                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <h3>Predicted Emotion: <span class="{emotion_classes[emotion]}">{emotion_icons[emotion]} {emotion.capitalize()}</span></h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if show_details:
                        st.markdown("### Detailed Emotion Scores")
                        scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=['Score'])
                        scores_df = scores_df.sort_values('Score', ascending=False)

                        # Table
                        st.dataframe(scores_df.style.highlight_max(axis=0), height=200)

                        # Chart
                        st.markdown("#### Emotion Scores Chart")
                        st.bar_chart(scores_df)

                        # HTML Display
                        for emotion_item, score in scores_df.itertuples():
                            st.markdown(
                                f"""
                                <div class="prediction-box">
                                    <p>{emotion_icons[emotion_item]} <strong>{emotion_item.capitalize()}</strong></p>
                                    <p class="emotion-score">Score: {score:.2f}</p>
                                    <progress value="{score}" max="{scores_df['Score'].max() + 1}"></progress>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

    with col2:
        st.subheader("Example Texts")
        examples = {
            "😊 Happy": "Saya sangat senang dengan pelayanan yang diberikan, sangat memuaskan!",
            "😍 Love": "Produk ini luar biasa, saya jatuh cinta sejak pertama kali mencoba.",
            "😠 Anger": "Saya sangat kecewa dengan barang ini, kualitas sangat buruk!",
            "😢 Sadness": "Sedih sekali barangnya tidak sesuai dengan harapan saya.",
            "😨 Fear": "Saya khawatir produk ini tidak aman untuk digunakan."
        }

        for emotion, text in examples.items():
            if st.button(f"{emotion}: {text[:30]}..."):
                st.session_state["user_input"] = text  # save in session to use in text area

        # Optional: set text area to use session state
        if "user_input" in st.session_state:
            st.text_area("Input", value=st.session_state["user_input"], height=150, key="user_input_area")

if __name__ == "__main__":
    main()
