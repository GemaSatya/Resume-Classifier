# -*- coding: utf-8 -*-
"""
train_and_save.py
=================
Jalankan script ini SEKALI SAJA di lokal atau di Streamlit Cloud via terminal:
    python train_and_save.py

Script ini akan melatih semua model dan menyimpan hasilnya ke folder `model_cache/`.
Setelah itu, main.py akan memuat model dari cache tanpa perlu melatih ulang.
"""

import os, re, string, joblib, json
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ── Setup ─────────────────────────────────────────────────────
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

stop_words  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    words = [lemmatizer.lemmatize(w) for w in text.split()
             if w not in stop_words and len(w) > 2]
    return " ".join(words)

# ── Load dataset ──────────────────────────────────────────────
CSV = "UpdatedResumeDataSet.csv"
if not os.path.exists(CSV):
    raise FileNotFoundError(f"File {CSV} tidak ditemukan. Letakkan di folder yang sama.")

print(f"[1/6] Membaca dataset: {CSV}")
df = pd.read_csv(CSV)
df["Cleaned"] = df["Resume"].apply(clean_text)
X_raw, y = df["Cleaned"], df["Category"]

# ── Split ─────────────────────────────────────────────────────
print("[2/6] Membagi data train/test (80/20) ...")
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y)

# ── Vectorizer ────────────────────────────────────────────────
print("[3/6] Melatih TF-IDF vectorizer ...")
vec = TfidfVectorizer(max_features=1000, ngram_range=(1,1),
                      min_df=5, max_df=0.80, sublinear_tf=True)
X_tr = vec.fit_transform(X_tr_raw)
X_te = vec.transform(X_te_raw)

enc = LabelEncoder()
y_tr_enc = enc.fit_transform(y_tr)
y_te_enc = enc.transform(y_te)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Training helper ───────────────────────────────────────────
def fit_eval(name, mdl, use_enc=False):
    print(f"    Melatih {name} ...")
    if use_enc:
        mdl.fit(X_tr, y_tr_enc)
        pred_enc = mdl.predict(X_te)
        pred = enc.inverse_transform(pred_enc)
        cv_s = cross_val_score(mdl, X_tr, y_tr_enc, cv=cv5, scoring="accuracy")
        tr_acc = accuracy_score(y_tr, enc.inverse_transform(mdl.predict(X_tr)))
    else:
        mdl.fit(X_tr, y_tr)
        pred = mdl.predict(X_te)
        cv_s = cross_val_score(mdl, X_tr, y_tr, cv=cv5, scoring="accuracy")
        tr_acc = accuracy_score(y_tr, mdl.predict(X_tr))

    classes_sorted = sorted(y_te.unique())
    return {
        "model":      mdl,
        "train_acc":  round(tr_acc, 4),
        "cv_mean":    round(float(cv_s.mean()), 4),
        "cv_std":     round(float(cv_s.std()),  4),
        "test_acc":   round(accuracy_score(y_te, pred), 4),
        "precision":  round(precision_score(y_te, pred, average="weighted", zero_division=0), 4),
        "recall":     round(recall_score(y_te, pred, average="weighted", zero_division=0), 4),
        "f1":         round(f1_score(y_te, pred, average="weighted", zero_division=0), 4),
        "cm":         confusion_matrix(y_te, pred, labels=classes_sorted).tolist(),  # list for JSON
        "report":     classification_report(y_te, pred, zero_division=0),
    }

# ── Train all ─────────────────────────────────────────────────
print("[4/6] Melatih semua model ...")
bundle = {
    "Decision Tree": fit_eval("Decision Tree", DecisionTreeClassifier(
        criterion="entropy", max_depth=12, min_samples_split=4,
        min_samples_leaf=2, class_weight="balanced", random_state=42)),
    "Random Forest": fit_eval("Random Forest", RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_split=15,
        min_samples_leaf=8, max_features="sqrt", random_state=42, n_jobs=-1)),
    "SVM": fit_eval("SVM", SVC(
        kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    "XGBoost": fit_eval("XGBoost", XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.6, colsample_bytree=0.6, reg_alpha=2.0, reg_lambda=5.0,
        min_child_weight=12, random_state=42, eval_metric="mlogloss", verbosity=0),
        use_enc=True),
}

# ── Simpan ke disk ────────────────────────────────────────────
print("[5/6] Menyimpan model ke folder model_cache/ ...")
os.makedirs("model_cache", exist_ok=True)

# Simpan vectorizer dan encoder
joblib.dump(vec, "model_cache/vectorizer.pkl")
joblib.dump(enc, "model_cache/label_encoder.pkl")

# Simpan setiap model dan metrik-nya secara terpisah
classes_sorted = sorted(y_te.unique())
for name, data in bundle.items():
    safe_name = name.lower().replace(" ", "_")
    # Simpan objek model
    joblib.dump(data["model"], f"model_cache/{safe_name}.pkl")
    # Simpan metrik (tanpa objek model) sebagai JSON agar ringan
    metrics = {k: v for k, v in data.items() if k != "model"}
    with open(f"model_cache/{safe_name}_metrics.json", "w") as f:
        json.dump(metrics, f)

# Simpan daftar kelas
with open("model_cache/classes.json", "w") as f:
    json.dump(classes_sorted, f)

print("[6/6] Selesai! File yang tersimpan di model_cache/:")
for fname in sorted(os.listdir("model_cache")):
    size_kb = os.path.getsize(f"model_cache/{fname}") / 1024
    print(f"    {fname:45s} {size_kb:8.1f} KB")

print("\n✅ Semua model berhasil disimpan. Sekarang jalankan main.py!")