# -*- coding: utf-8 -*-
"""
Aplikasi Streamlit — Resume Classification Form
Komparasi: Decision Tree | Random Forest | SVM | XGBoost

CARA DEPLOY:
1. Jalankan `python train_and_save.py` sekali untuk membuat folder model_cache/
2. Commit folder model_cache/ ke repo GitHub bersama main.py
3. Deploy ke Streamlit Cloud — model langsung dimuat, tidak perlu latih ulang
"""

import streamlit as st
import pandas as pd
import numpy as np
import re, string, os, json

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import joblib

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Resume Classifier", page_icon="📄", layout="wide")

st.markdown("""
<style>
body { background: #f5f6fa; }
.section-title {
    font-size: 15px; font-weight: 700; color: #4C72B0;
    text-transform: uppercase; letter-spacing: 1px;
    margin: 18px 0 8px 0; border-bottom: 2px solid #e0e6f0; padding-bottom: 4px;
}
.result-card {
    border-radius: 12px; padding: 20px 18px 14px 18px;
    text-align: center; margin-bottom: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.result-card .model-name { font-size: 13px; font-weight: 700; color: #fff; opacity: 0.85; margin-bottom: 6px; }
.result-card .pred-label { font-size: 20px; font-weight: 800; color: #fff; line-height: 1.2; }
.result-card .conf { font-size: 12px; color: rgba(255,255,255,0.8); margin-top: 6px; }
.verdict-box {
    border-radius: 10px; padding: 16px 20px;
    font-size: 17px; font-weight: 700; text-align: center; margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── NLTK setup ───────────────────────────────────────────────
@st.cache_resource
def setup_nltk():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

setup_nltk()

@st.cache_resource
def get_preprocessor():
    return set(stopwords.words("english")), WordNetLemmatizer()

stop_words, lemmatizer = get_preprocessor()

def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    words = [lemmatizer.lemmatize(w) for w in text.split()
             if w not in stop_words and len(w) > 2]
    return " ".join(words)

# ── Load models dari cache ────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
MODEL_NAMES = ["Decision Tree", "Random Forest", "SVM", "XGBoost"]

@st.cache_resource(show_spinner=False)
def load_all_models():
    """Muat model dari file .pkl — jauh lebih cepat daripada melatih ulang."""
    if not os.path.exists(CACHE_DIR):
        return None, None, None, None

    vec = joblib.load(os.path.join(CACHE_DIR, "vectorizer.pkl"))
    enc = joblib.load(os.path.join(CACHE_DIR, "label_encoder.pkl"))

    with open(os.path.join(CACHE_DIR, "classes.json")) as f:
        classes = json.load(f)

    bundle = {}
    for name in MODEL_NAMES:
        safe_name = name.lower().replace(" ", "_")
        model_path   = os.path.join(CACHE_DIR, f"{safe_name}.pkl")
        metrics_path = os.path.join(CACHE_DIR, f"{safe_name}_metrics.json")

        mdl = joblib.load(model_path)
        with open(metrics_path) as f:
            metrics = json.load(f)

        # cm disimpan sebagai list, ubah kembali ke numpy array
        metrics["cm"] = np.array(metrics["cm"])
        metrics["model"] = mdl
        bundle[name] = metrics

    return vec, enc, bundle, classes

# ── Cek apakah model_cache ada ───────────────────────────────
cache_exists = os.path.exists(CACHE_DIR) and os.path.exists(
    os.path.join(CACHE_DIR, "vectorizer.pkl"))

if not cache_exists:
    st.error(
        "❌ Folder **model_cache/** tidak ditemukan atau kosong.\n\n"
        "**Langkah yang perlu dilakukan:**\n"
        "1. Pastikan `UpdatedResumeDataSet.csv` ada di folder yang sama\n"
        "2. Jalankan perintah berikut di terminal lokal:\n"
        "   ```\n   python train_and_save.py\n   ```\n"
        "3. Commit folder `model_cache/` ke repository GitHub\n"
        "4. Deploy ulang aplikasi ini ke Streamlit Cloud"
    )
    st.stop()

with st.spinner("⏳ Memuat model dari cache..."):
    vec, enc, bundle, classes = load_all_models()

if bundle is None:
    st.error("Gagal memuat model dari cache. Jalankan `python train_and_save.py` terlebih dahulu.")
    st.stop()

best_model = max(bundle, key=lambda m: bundle[m]["f1"])

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.title("📄 Resume Classification Comparator")
st.caption("Isi form CV di bawah → keempat model mengklasifikasikan secara bersamaan, lalu hasilnya dibandingkan")

tab_form, tab_compare, tab_viz = st.tabs(["📝 Form Input CV", "📊 Komparasi Model", "📈 Visualisasi"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — FORM INPUT CV
# ═══════════════════════════════════════════════════════════════
with tab_form:
    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown("### 🧑‍💼 Data Diri")
        c1, c2 = st.columns(2)
        with c1:
            nama    = st.text_input("Nama Lengkap",  placeholder="Budi Santoso")
            email   = st.text_input("Email",          placeholder="budi@email.com")
        with c2:
            telepon = st.text_input("No. Telepon",    placeholder="08xx-xxxx-xxxx")
            kota    = st.text_input("Kota / Lokasi",  placeholder="Semarang, Jawa Tengah")

        st.markdown('<div class="section-title">🎓 Pendidikan</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            jenjang     = st.selectbox("Jenjang Pendidikan", ["SMA/SMK","D3","S1","S2","S3"])
            jurusan     = st.text_input("Jurusan / Program Studi", placeholder="Teknik Informatika")
        with c4:
            universitas = st.text_input("Nama Institusi", placeholder="Universitas Diponegoro")
            tahun_lulus = st.text_input("Tahun Lulus",    placeholder="2022")

        st.markdown('<div class="section-title">🛠️ Keahlian (Skills)</div>', unsafe_allow_html=True)
        skills_teknis = st.text_area("Skills Teknis",
            placeholder="Python, Machine Learning, SQL, TensorFlow, Scikit-learn, Pandas, NLP...",
            height=90)
        skills_non = st.text_area("Skills Non-Teknis / Soft Skills",
            placeholder="Komunikasi, Kepemimpinan, Manajemen Proyek, Analisis Data...",
            height=70)

        st.markdown('<div class="section-title">💼 Pengalaman Kerja</div>', unsafe_allow_html=True)
        num_exp = st.number_input("Jumlah pengalaman", min_value=0, max_value=5, value=1, step=1)

        exp_texts = []
        for i in range(int(num_exp)):
            with st.expander(f"Pengalaman #{i+1}", expanded=(i == 0)):
                cx1, cx2 = st.columns(2)
                with cx1:
                    posisi     = st.text_input("Posisi / Jabatan",  key=f"pos_{i}",  placeholder="Data Scientist")
                    perusahaan = st.text_input("Nama Perusahaan",   key=f"co_{i}",   placeholder="PT. Teknologi Maju")
                with cx2:
                    periode    = st.text_input("Periode",            key=f"per_{i}",  placeholder="Jan 2021 – Des 2022")
                    lama       = st.text_input("Lama Bekerja",       key=f"lama_{i}", placeholder="2 tahun")
                deskripsi = st.text_area("Deskripsi Pekerjaan", key=f"desc_{i}",
                    placeholder="Mengembangkan model prediksi menggunakan Python dan Scikit-learn...",
                    height=80)
                exp_texts.append(f"{posisi} at {perusahaan} {periode} {lama}. {deskripsi}")

        st.markdown('<div class="section-title">📜 Sertifikasi & Proyek</div>', unsafe_allow_html=True)
        sertifikasi = st.text_area("Sertifikasi",
            placeholder="TensorFlow Developer Certificate, AWS Cloud Practitioner, PMP...", height=70)
        proyek = st.text_area("Proyek yang Pernah Dikerjakan",
            placeholder="Sistem rekomendasi collaborative filtering; Aplikasi NLP analisis sentimen...", height=80)

        st.markdown('<div class="section-title">📌 Informasi Tambahan</div>', unsafe_allow_html=True)
        bahasa  = st.text_input("Bahasa yang Dikuasai", placeholder="Indonesia (Native), Inggris (Aktif)")
        hobi    = st.text_input("Hobi / Minat",         placeholder="Open-source contribution, competitive programming")
        summary = st.text_area("Ringkasan Profil (opsional)",
            placeholder="Profesional IT berpengalaman 3 tahun di bidang Data Science...", height=80)

        submitted = st.button("🚀 Klasifikasikan Resume Saya", type="primary", use_container_width=True)

    # ── KOLOM HASIL ──────────────────────────────────────────
    with col_result:
        st.markdown("### 🏁 Hasil Klasifikasi")

        if not submitted:
            st.info("Isi form di sebelah kiri, lalu klik **Klasifikasikan Resume Saya**.")
            st.markdown("**Kategori yang tersedia:**")
            cat_list = [
                "Data Science","Java Developer","Python Developer","Web Designing",
                "DevOps Engineer","Testing","Automation Testing","Network Security Engineer",
                "Database","Hadoop","ETL Developer","Blockchain","SAP Developer",
                "DotNet Developer","HR","Sales","Business Analyst","Operations Manager",
                "PMO","Mechanical Engineer","Civil Engineer","Electrical Engineering",
                "Health and fitness","Arts","Advocate",
            ]
            cols3 = st.columns(2)
            for i, c in enumerate(cat_list):
                cols3[i % 2].markdown(f"• {c}")

        else:
            resume_parts = []
            if summary:       resume_parts.append(f"Profile: {summary}")
            if jurusan:       resume_parts.append(f"Education: {jenjang} {jurusan} {universitas} {tahun_lulus}")
            if skills_teknis: resume_parts.append(f"Technical Skills: {skills_teknis}")
            if skills_non:    resume_parts.append(f"Skills: {skills_non}")
            for e in exp_texts:
                if e.strip(): resume_parts.append(f"Experience: {e}")
            if sertifikasi:   resume_parts.append(f"Certifications: {sertifikasi}")
            if proyek:        resume_parts.append(f"Projects: {proyek}")
            if bahasa:        resume_parts.append(f"Languages: {bahasa}")
            if hobi:          resume_parts.append(f"Interests: {hobi}")

            full_resume = " ".join(resume_parts)

            if len(full_resume.strip()) < 20:
                st.error("Isi minimal beberapa field agar model dapat mengklasifikasikan.")
            else:
                cleaned = clean_text(full_resume)
                vect    = vec.transform([cleaned])

                colors = {
                    "Decision Tree": "#4C72B0",
                    "Random Forest": "#2d8a4e",
                    "SVM":           "#C44E52",
                    "XGBoost":       "#7B5EA7",
                }
                icons = {
                    "Decision Tree": "🌳",
                    "Random Forest": "🌲",
                    "SVM":           "🔵",
                    "XGBoost":       "⚡",
                }

                preds = {}
                confs = {}
                for mname, mdata in bundle.items():
                    mdl = mdata["model"]
                    if mname == "XGBoost":
                        pred_label = enc.inverse_transform(mdl.predict(vect))[0]
                        proba      = mdl.predict_proba(vect)[0]
                    else:
                        pred_label = mdl.predict(vect)[0]
                        proba      = mdl.predict_proba(vect)[0]
                    preds[mname] = pred_label
                    confs[mname] = round(float(proba.max() * 100), 1)

                c_dt, c_rf = st.columns(2)
                c_sv, c_xg = st.columns(2)

                def render_card(col, mname):
                    label = preds[mname]
                    conf  = confs[mname]
                    clr   = colors[mname]
                    icon  = icons[mname]
                    badge = " 🏆" if mname == best_model else ""
                    with col:
                        st.markdown(
                            f"<div class='result-card' style='background:linear-gradient(135deg,{clr}dd,{clr});'>"
                            f"<div class='model-name'>{icon} {mname}{badge}</div>"
                            f"<div class='pred-label'>{label}</div>"
                            f"<div class='conf'>Confidence {conf}%</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.progress(int(conf))

                render_card(c_dt, "Decision Tree")
                render_card(c_rf, "Random Forest")
                render_card(c_sv, "SVM")
                render_card(c_xg, "XGBoost")

                vote      = pd.Series(list(preds.values())).value_counts()
                top_cat   = vote.index[0]
                top_count = vote.iloc[0]

                st.markdown("---")
                if top_count == 4:
                    st.markdown(
                        f"<div class='verdict-box' style='background:#d4edda;color:#155724;'>"
                        f"✅ Semua model sepakat: <b>{top_cat}</b></div>",
                        unsafe_allow_html=True)
                elif top_count >= 3:
                    st.markdown(
                        f"<div class='verdict-box' style='background:#fff3cd;color:#856404;'>"
                        f"⚠️ Mayoritas model ({top_count}/4): <b>{top_cat}</b></div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div class='verdict-box' style='background:#f8d7da;color:#721c24;'>"
                        f"❓ Model tidak sepakat — hasil bervariasi</div>",
                        unsafe_allow_html=True)

                st.markdown("**Detail Confidence per Model:**")
                st.dataframe(pd.DataFrame({
                    "Model":      list(preds.keys()),
                    "Prediksi":   list(preds.values()),
                    "Confidence": [f"{confs[m]}%" for m in preds],
                }), hide_index=True, use_container_width=True)

                with st.expander("📋 Teks resume yang diproses model"):
                    st.text_area("", value=full_resume, height=150, disabled=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — KOMPARASI MODEL
# ═══════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("📊 Tabel Perbandingan Metrik (pada dataset uji)")

    df_res = pd.DataFrame([{
        "Model":      name,
        "Train Acc":  d["train_acc"],
        "CV Acc":     d["cv_mean"],
        "CV Std (±)": d["cv_std"],
        "Test Acc":   d["test_acc"],
        "Precision":  d["precision"],
        "Recall":     d["recall"],
        "F1-Score":   d["f1"],
    } for name, d in bundle.items()])

    def hl(col):
        if col.name == "CV Std (±)":
            best = col.min()
            return ["background-color:#d4edda;font-weight:700" if v == best else "" for v in col]
        if col.name == "Model":
            return [""] * len(col)
        best = col.max()
        return ["background-color:#d4edda;font-weight:700" if v == best else "" for v in col]

    st.dataframe(
        df_res.style.apply(hl).format({c: "{:.4f}" for c in df_res.columns if c != "Model"}),
        use_container_width=True, hide_index=True,
    )
    st.caption("🟢 Hijau = nilai terbaik per kolom. CV Std: lebih kecil lebih stabil.")

    st.markdown("---")
    st.subheader("Analisis Generalisasi Model")
    for name, d in bundle.items():
        gap = d["train_acc"] - d["test_acc"]
        if gap > 0.15:
            icon, label, clr = "⚠️", f"Overfitting (gap Train–Test: {gap:.4f})", "orange"
        elif d["test_acc"] < 0.5:
            icon, label, clr = "❌", f"Underfitting (Test Acc rendah: {d['test_acc']:.4f})", "red"
        else:
            icon, label, clr = "✅", f"Generalisasi baik (gap: {gap:.4f})", "green"
        st.markdown(f"**{name}** — <span style='color:{clr}'>{icon} {label}</span>",
                    unsafe_allow_html=True)

    with st.expander("📋 Classification Report per Model"):
        for name, d in bundle.items():
            st.markdown(f"**{name}**")
            st.code(d["report"], language="text")

# ═══════════════════════════════════════════════════════════════
# TAB 3 — VISUALISASI
# ═══════════════════════════════════════════════════════════════
with tab_viz:
    pal   = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    names = list(bundle.keys())

    st.subheader("Perbandingan Metrik Test Set")
    fig1, ax1 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names)); w = 0.18
    for i, (key, lbl) in enumerate([("test_acc","Test Acc"),("precision","Precision"),
                                     ("recall","Recall"),("f1","F1-Score")]):
        ax1.bar(x + i*w, [bundle[m][key] for m in names], w, label=lbl, alpha=0.85)
    ax1.set_xticks(x + w*1.5); ax1.set_xticklabels(names)
    ax1.set_ylim(0, 1.15); ax1.set_ylabel("Score")
    ax1.set_title("Perbandingan Metrik Evaluasi (Test Set)")
    ax1.legend(); ax1.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig1); plt.close(fig1)

    st.subheader("Train vs CV vs Test Accuracy")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    ax2.bar(x-w, [bundle[m]["train_acc"] for m in names], w, label="Train", color="#4C72B0", alpha=0.85)
    ax2.bar(x,   [bundle[m]["cv_mean"]   for m in names], w, label="CV",    color="#55A868", alpha=0.85)
    ax2.bar(x+w, [bundle[m]["test_acc"]  for m in names], w, label="Test",  color="#C44E52", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(names)
    ax2.set_ylim(0, 1.15); ax2.legend(); ax2.set_ylabel("Accuracy")
    ax2.set_title("Train vs CV vs Test Accuracy")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    st.subheader("Cross-Validation Accuracy (5-Fold Stratified)")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    cv_m = [bundle[m]["cv_mean"] for m in names]
    cv_s = [bundle[m]["cv_std"]  for m in names]
    bars = ax3.bar(names, cv_m, yerr=cv_s, capsize=6, color=pal, alpha=0.85, ecolor="black")
    for bar, val in zip(bars, cv_m):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax3.set_ylim(0, 1.15); ax3.set_ylabel("CV Accuracy")
    ax3.set_title("Cross-Validation Accuracy ± Std Dev")
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

    st.subheader("Confusion Matrix — Semua Model")
    fig4, axes = plt.subplots(2, 2, figsize=(20, 16))
    for ax, name in zip(axes.flatten(), names):
        sns.heatmap(bundle[name]["cm"], annot=False, cmap="Blues", ax=ax,
                    xticklabels=classes, yticklabels=classes)
        ax.set_title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", rotation=0,  labelsize=6)
    plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)