from flask import Flask, render_template, request
import tensorflow as tf
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)
app = Flask(__name__)

# Load model + tokenizer
model = tf.keras.models.load_model("model.h5", compile=False)
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

def predict_news(text):
    if text.strip() == "":
        return "Please enter valid text", ""
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=500)
    pred = model.predict(padded)[0][0]
    if pred > 0.6:
        confidence = round(pred * 100, 2)
        return "Real News 🟢", confidence
    else:
        confidence = round((1 - pred) * 100, 2)
        return "Fake News 🔴", confidence
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    confidence = ""
    table = None
    if request.method == "POST":
        # TEXT INPUT
        if "news" in request.form and request.form["news"] != "":
            news = request.form["news"]
            result, confidence = predict_news(news)
        #FILE UPLOAD
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                df = pd.read_csv(file)
                # assume column name is 'text'
                if "text" in df.columns:
                    df["prediction"] = df["text"].apply(lambda x: predict_news(x)[0])
                    table = df.head(10).to_html(classes='table table-striped')
    return render_template("index.html", result=result, confidence=confidence, table=table)

import os

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))