"""
Sentiment Analysis & Fake Review Detection — Flask Backend
Algorithm : Bidirectional LSTM + Word Embeddings (Deep Learning)
Dataset   : 12,000 labeled reviews from CSV
Cache     : Trained models saved to disk — fast restarts after first run.
"""

import os, csv, re, pickle, logging
import numpy as np
import mysql.connector
from collections import Counter

# ── Suppress verbose TF logs ──────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from flask import Flask, render_template, request, jsonify

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

app = Flask(__name__)

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
MAX_WORDS     = 12000   # vocabulary size
MAX_LEN       = 120     # max tokens per review
EMBEDDING_DIM = 64      # word vector size
LSTM_UNITS    = 64      # LSTM hidden units
EPOCHS        = 15
BATCH_SIZE    = 32

# ─── Model cache paths ─────────────────────────────────────────────────────────
MODEL_DIR           = os.path.join(os.path.dirname(__file__), 'models')
SENT_MODEL_PATH     = os.path.join(MODEL_DIR, 'sentiment_lstm.keras')
FAKE_MODEL_PATH     = os.path.join(MODEL_DIR, 'fake_lstm.keras')
TOKENIZER_PATH      = os.path.join(MODEL_DIR, 'tokenizer.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Database Configuration ────────────────────────────────────────────────────
DB_CONFIG = {
    'host':     'localhost',
    'user':     'root',
    'password': 'mathanmani@123',
    'database': 'fake_review_db'
}

def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        print(f"[DB-ERROR] {err}")
        return None

# ─── Load CSV Dataset ──────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), 'reviews (1).csv')

_sent_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
_fake_map  = {'Genuine': 0, 'Fake': 1}

sentiment_texts, sentiment_labels = [], []
fake_texts,      fake_labels      = [], []
ALL_REVIEWS = []

print("[*] Loading CSV dataset...")
try:
    with open(CSV_PATH, encoding='utf-8', errors='ignore') as f:
        for row in csv.DictReader(f):
            review = row.get('review', '').strip()
            label  = row.get('label',  '').strip()
            sent   = row.get('sentiment', '').strip()
            if not review:
                continue
            ALL_REVIEWS.append({'review': review, 'label': label, 'sentiment': sent})
            if sent in _sent_map:
                sentiment_texts.append(review)
                sentiment_labels.append(_sent_map[sent])
            if label in _fake_map:
                fake_texts.append(review)
                fake_labels.append(_fake_map[label])
    print(f"[OK] Loaded {len(ALL_REVIEWS)} reviews "
          f"| Sentiment: {len(sentiment_texts)} | Fake: {len(fake_texts)}")
except FileNotFoundError:
    print(f"[WARN] CSV not found — using minimal fallback data")
    sentiment_texts  = ["great product", "terrible product", "okay product"]
    sentiment_labels = [2, 0, 1]
    fake_texts  = ["nice buy", "BUY NOW BUY NOW!!!"]
    fake_labels = [0, 1]


# ─── Text Preprocessing ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

all_texts_clean = [clean_text(t) for t in sentiment_texts + fake_texts]


# ─── Build / Load Tokenizer ────────────────────────────────────────────────────
if os.path.exists(TOKENIZER_PATH):
    print("[*] Loading tokenizer from cache...")
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("[OK] Tokenizer loaded")
else:
    print("[*] Building tokenizer...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(all_texts_clean)
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"[OK] Tokenizer built — vocab size: {len(tokenizer.word_index)}")


def texts_to_padded(texts):
    cleaned = [clean_text(t) for t in texts]
    seqs    = tokenizer.texts_to_sequences(cleaned)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')


# ─── LSTM Architecture Builder ─────────────────────────────────────────────────
def build_lstm_model(num_classes: int) -> tf.keras.Model:
    """
    Bidirectional LSTM model:
      Embedding → BiLSTM → GlobalMaxPool → Dropout → Dense → Output
    """
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes,
              activation='softmax' if num_classes > 1 else 'sigmoid'),
    ])
    loss = 'categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model


early_stop = EarlyStopping(monitor='val_loss', patience=3,
                            restore_best_weights=True, verbose=0)


# ─── Sentiment LSTM ────────────────────────────────────────────────────────────
if os.path.exists(SENT_MODEL_PATH):
    print("[*] Loading Sentiment LSTM from cache...")
    sentiment_model = load_model(SENT_MODEL_PATH)
    print("[OK] Sentiment LSTM loaded")
else:
    print("[*] Training Sentiment LSTM (Bidirectional) — this may take a few minutes...")
    X_sent = texts_to_padded(sentiment_texts)
    y_sent = to_categorical(sentiment_labels, num_classes=3)

    sentiment_model = build_lstm_model(num_classes=3)
    sentiment_model.fit(
        X_sent, y_sent,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )
    sentiment_model.save(SENT_MODEL_PATH)
    print("[OK] Sentiment LSTM trained & saved")


# ─── Fake-Detection LSTM ───────────────────────────────────────────────────────
if os.path.exists(FAKE_MODEL_PATH):
    print("[*] Loading Fake-Detection LSTM from cache...")
    fake_model = load_model(FAKE_MODEL_PATH)
    print("[OK] Fake-Detection LSTM loaded")
else:
    print("[*] Training Fake-Detection LSTM (Bidirectional) — this may take a few minutes...")
    X_fake = texts_to_padded(fake_texts)
    y_fake = to_categorical(fake_labels, num_classes=2)

    fake_model = build_lstm_model(num_classes=2)
    fake_model.fit(
        X_fake, y_fake,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )
    fake_model.save(FAKE_MODEL_PATH)
    print("[OK] Fake-Detection LSTM trained & saved")


# ─── Heuristic Fake Signals ────────────────────────────────────────────────────
def _has_repetition(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 4:
        return False
    counts = Counter(words)
    return counts.most_common(1)[0][1] / len(words) > 0.4

FAKE_SIGNALS = {
    "excessive_caps":        (lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1) > 0.5,
                              "Excessive use of capital letters"),
    "excessive_exclamation": (lambda t: t.count("!") > 3,
                              "Excessive exclamation marks"),
    "repetitive_words":      (lambda t: _has_repetition(t),
                              "Highly repetitive language"),
    "very_short":            (lambda t: len(t.split()) < 4,
                              "Review is suspiciously short"),
    "buy_now_pressure":      (lambda t: any(p in t.lower() for p in
                              ["buy now", "buy it now", "buy immediately", "must buy"]),
                              "Contains aggressive purchase pressure language"),
    "incentivised":          (lambda t: any(p in t.lower() for p in
                              ["discount for review", "free product", "asked me to leave"]),
                              "Appears to be an incentivised review"),
}

def heuristic_fake_score(text: str):
    reasons, score = [], 0.0
    for _, (fn, reason) in FAKE_SIGNALS.items():
        if fn(text):
            reasons.append(reason)
            score += 0.15
    return min(score, 0.6), reasons


# ─── Helper Functions ──────────────────────────────────────────────────────────
def confidence_to_stars(pos: float, neg: float, neu: float) -> float:
    raw = pos * 5.0 + neu * 3.0 + neg * 1.0
    return round(max(1.0, min(5.0, raw)), 1)

def sentiment_reason(label: str, confidence: float) -> str:
    strength = "strongly" if confidence > 0.80 else "moderately" if confidence > 0.55 else "slightly"
    if label == "Positive":
        return f"The review expresses {strength} positive sentiment with favorable language and tone."
    elif label == "Negative":
        return f"The review expresses {strength} negative sentiment indicating dissatisfaction."
    return f"The review is {strength} neutral, expressing neither strong praise nor criticism."


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = data.get("review", "").strip()
    if not text:
        return jsonify({"error": "Review text is required."}), 400

    padded = texts_to_padded([text])

    # ── Sentiment LSTM prediction ──────────────────────────────────────────
    sent_proba = sentiment_model.predict(padded, verbose=0)[0]
    # Index order matches label encoding: 0=Negative, 1=Neutral, 2=Positive
    neg_conf = float(sent_proba[0])
    neu_conf = float(sent_proba[1])
    pos_conf = float(sent_proba[2])

    sent_idx       = int(np.argmax(sent_proba))
    sent_label     = ["Negative", "Neutral", "Positive"][sent_idx]
    sent_confidence = float(sent_proba[sent_idx])

    # ── Fake-Detection LSTM prediction ────────────────────────────────────
    fake_proba       = fake_model.predict(padded, verbose=0)[0]
    dl_genuine_conf  = float(fake_proba[0])
    dl_fake_conf     = float(fake_proba[1])

    # ── Heuristic boost ────────────────────────────────────────────────────
    h_boost, h_reasons      = heuristic_fake_score(text)
    combined_fake_conf      = min(dl_fake_conf + h_boost, 1.0)
    combined_genuine_conf   = max(1.0 - combined_fake_conf, 0.0)

    is_fake        = combined_fake_conf > 0.5
    fake_label     = "Fake" if is_fake else "Genuine"
    fake_confidence = combined_fake_conf if is_fake else combined_genuine_conf

    if is_fake:
        fake_reason = ("Flagged as potentially fake: " + "; ".join(h_reasons) + "."
                       if h_reasons
                       else "The LSTM model detected patterns commonly seen in fake reviews.")
    else:
        fake_reason = "The review appears authentic with natural language patterns and specific details."

    stars = confidence_to_stars(pos_conf, neg_conf, neu_conf)

    # ── Database insertion ─────────────────────────────────────────────────
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO reviews
                   (review_text, sentiment_label, sentiment_confidence,
                    fake_label, fake_confidence, stars)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (text, sent_label, float(sent_confidence),
                 fake_label, float(fake_confidence), float(stars))
            )
            conn.commit()
            cursor.close()
            conn.close()
    except Exception as e:
        print(f"[DB-ERROR] {e}")

    return jsonify({
        "sentiment": {
            "label":      sent_label,
            "confidence": round(sent_confidence * 100, 2),
            "reason":     sentiment_reason(sent_label, sent_confidence),
        },
        "fake_detection": {
            "label":      fake_label,
            "confidence": round(fake_confidence * 100, 2),
            "reason":     fake_reason,
        },
        "scores": {
            "positive": round(pos_conf * 100, 2),
            "negative": round(neg_conf * 100, 2),
            "neutral":  round(neu_conf * 100, 2),
            "fake":     round(combined_fake_conf * 100, 2),
            "genuine":  round(combined_genuine_conf * 100, 2),
        },
        "stars": stars,
    })


# ─── Dataset Routes ────────────────────────────────────────────────────────────
@app.route("/dataset")
def dataset_page():
    return render_template("dataset.html")


@app.route("/api/dataset")
def api_dataset():
    page     = max(int(request.args.get('page', 1)), 1)
    per_page = min(int(request.args.get('per_page', 20)), 100)
    search   = request.args.get('search', '').lower().strip()
    label_f  = request.args.get('label', '').strip()
    sent_f   = request.args.get('sentiment', '').strip()

    filtered = ALL_REVIEWS
    if search:
        filtered = [r for r in filtered if search in r['review'].lower()]
    if label_f:
        filtered = [r for r in filtered if r['label'] == label_f]
    if sent_f:
        filtered = [r for r in filtered if r['sentiment'] == sent_f]

    total = len(filtered)
    rows  = filtered[(page-1)*per_page : page*per_page]

    return jsonify({
        'total':    total,
        'page':     page,
        'per_page': per_page,
        'pages':    max(1, -(-total // per_page)),
        'reviews':  rows,
    })


@app.route("/api/dataset-stats")
def api_dataset_stats():
    return jsonify({
        'total':    len(ALL_REVIEWS),
        'genuine':  sum(1 for r in ALL_REVIEWS if r['label']     == 'Genuine'),
        'fake':     sum(1 for r in ALL_REVIEWS if r['label']     == 'Fake'),
        'positive': sum(1 for r in ALL_REVIEWS if r['sentiment'] == 'Positive'),
        'negative': sum(1 for r in ALL_REVIEWS if r['sentiment'] == 'Negative'),
        'neutral':  sum(1 for r in ALL_REVIEWS if r['sentiment'] == 'Neutral'),
    })


# ─── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n>>> Server running at http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)
