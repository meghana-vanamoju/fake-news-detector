import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
from numpy_model import NumpyModel

# Set page config first
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_resources():
    import sys
    import types
    
    # Create mock modules for keras.src.legacy so the tokenizer pickle can load
    legacy_mod = types.ModuleType('keras.src.legacy')
    preprocessing_mod = types.ModuleType('keras.src.legacy.preprocessing')
    text_mod = types.ModuleType('keras.src.legacy.preprocessing.text')
    
    # We need the Tokenizer class - define a minimal one if keras isn't available
    try:
        import keras.preprocessing.text
        text_mod.Tokenizer = keras.preprocessing.text.Tokenizer
    except ImportError:
        # Fallback: the pickle will bring its own class definition
        pass
    
    legacy_mod.preprocessing = preprocessing_mod
    preprocessing_mod.text = text_mod
    sys.modules['keras'] = sys.modules.get('keras', types.ModuleType('keras'))
    sys.modules['keras.src'] = types.ModuleType('keras.src')
    sys.modules['keras.src.legacy'] = legacy_mod
    sys.modules['keras.src.legacy.preprocessing'] = preprocessing_mod
    sys.modules['keras.src.legacy.preprocessing.text'] = text_mod
    
    nltk.download('stopwords', quiet=True)
    
    # Load numpy-based model (no TensorFlow needed!)
    model = NumpyModel("model_weights.npz")
    
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    stop_words = set(stopwords.words('english'))
    return model, tokenizer, stop_words

# Display a loading message while models are loaded
with st.spinner("Loading models..."):
    model, tokenizer, stop_words = load_resources()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

def pad_sequences_np(sequences, maxlen):
    result = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            result[i] = np.array(seq[-maxlen:])
        else:
            result[i, maxlen - len(seq):] = np.array(seq)
    return result

def predict_news(text):
    if text.strip() == "":
        return None, None, None
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences_np(seq, maxlen=500)
    pred_real = float(model.predict(padded)[0][0])
    pred_fake = 1.0 - pred_real
    
    label = "Real News" if pred_real > 0.6 else "Fake News"
    return label, pred_real, pred_fake

st.markdown("""
<style>
    .title-text {
        font-family: 'Inter', sans-serif;
        text-align: center;
        font-weight: 800;
        margin-bottom: 5px;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle-text {
        color: #6c757d;
        font-family: 'Inter', sans-serif;
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .stProgress .st-bo {
        background-color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-text'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Detect misinformation using deep learning</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["✍️ Text Analysis", "📁 Batch Processing (CSV)"])

with tab1:
    st.subheader("Analyze Article Text")
    news_text = st.text_area("Paste the news article text here:", height=250, placeholder="Enter the news content you want to verify...")
    
    if st.button("Detect Fake News", use_container_width=True, type="primary"):
        if news_text.strip():
            with st.spinner("Analyzing text..."):
                label, pred_real, pred_fake = predict_news(news_text)
                st.write("---")
                if label == "Real News":
                    st.success(f"### 🟢 {label}")
                else:
                    st.error(f"### 🔴 {label}")
                
                st.markdown("#### Confidence Breakdown:")
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown("**Real News:**")
                with col2:
                    st.progress(pred_real, text=f"{pred_real*100:.1f}%")

                col3, col4 = st.columns([1, 4])
                with col3:
                    st.markdown("**Fake News:**")
                with col4:
                    st.progress(pred_fake, text=f"{pred_fake*100:.1f}%")
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.subheader("Batch Process Articles")
    st.markdown("Upload a CSV file containing a `text` column to analyze multiple articles at once.")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "text" in df.columns.str.lower():
                text_col = [col for col in df.columns if col.lower() == 'text'][0]
                with st.spinner("Processing documents..."):
                    predictions = df[text_col].apply(lambda x: predict_news(x) if isinstance(x, str) else (None, None, None))
                    df["prediction"] = [p[0] for p in predictions]
                    df["confidence_real"] = [p[1] for p in predictions]
                    df["confidence_fake"] = [p[2] for p in predictions]
                    
                    st.success("Processing complete!")
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv,
                        file_name='fake_news_predictions.csv',
                        mime='text/csv',
                        type="primary"
                    )
            else:
                st.error("The CSV file must contain a 'text' column.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
