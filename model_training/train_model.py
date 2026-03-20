import pandas as pd

# 1. Load data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# LIAR dataset
liar = pd.read_csv("train.tsv", sep="\t", header=None)
liar = liar[[1, 2]]
liar.columns = ["label", "text"]

def convert_label(x):
    if x in ["true", "mostly-true"]:
        return 1
    else:
        return 0

liar["label"] = liar["label"].apply(convert_label)

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine
data = pd.concat([fake, true, liar])
data = data.sample(frac=1).reset_index(drop=True)

data = data[["text", "label"]]

# CLEANING
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

data["text"] = data["text"].apply(clean_text)

# TOKENIZATION
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data["text"])

X = tokenizer.texts_to_sequences(data["text"])
X = pad_sequences(X, maxlen=500)

y = data["label"]

# SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

print("Data ready for training ✅")

# MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential()

model.add(Embedding(input_dim=10000, output_dim=128, input_length=500))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

print("Model built successfully ✅")

# TRAIN
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=32,
    validation_data=(X_test, y_test)
)

print("Model training completed ✅")

# SAVE
model.save("model.h5")

import pickle
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))

print("Model saved successfully ✅")