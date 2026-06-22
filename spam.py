import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Nama file sesuai petunjuk tugas
DATASET_FILE = 'SMSSpamCollection.txt'
MODEL_FILE = 'model.pkl'
VECTORIZER_FILE = 'vectorizer.pkl'

# --- 1. PROSES TRAIN MODEL (Berjalan lokal jika file .pkl belum ada) ---
if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    if os.path.exists(DATASET_FILE):
        # Membaca dataset SMSSpamCollection (format TSV: label dan message)
        df = pd.read_csv(DATASET_FILE, sep='\t', names=['label', 'message'])
        
        # Proses Vectorizer
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['message'])
        y = df['label']
        
        # Split & Fit Model (Menggunakan Naive Bayes)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        # Menyimpan hasil train ke file .pkl (JANGAN DOWNLOAD, hasilkan sendiri!)
        joblib.dump(model, MODEL_FILE)
        joblib.dump(vectorizer, VECTORIZER_FILE)
    else:
        st.error(f"File dataset '{DATASET_FILE}' tidak ditemukan di folder ini! Silakan letakkan dataset di folder yang sama.")

# --- 2. LOAD MODEL DAN VECTORIZER ---
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
else:
    st.stop()

# --- 3. TAMPILAN INTERFACE STREAMLIT (Sesuai image_201bc4.png) ---
st.title("📩 SMS Spam Detection")
st.subheader("Cek apakah SMS termasuk spam:")

# Input Text Area
user_input = st.text_area("Masukkan pesan SMS:", height=100)

# Tombol Deteksi
if st.button("Deteksi"):
    if user_input.strip() != "":
        # Transformasi input teks user
        data_vector = vectorizer.transform([user_input])
        prediction = model.predict(data_vector)[0]
        
        # Hitung Probabilitas Spam
        classes = list(model.classes_)
        spam_index = classes.index('spam')
        proba_spam = model.predict_proba(data_vector)[0][spam_index]
        
        # Tampilkan Hasil (Catatan: Kedua kondisi menggunakan box sukses/hijau agar sesuai tampilan gambar)
        if prediction == 'spam':
            st.success("Hasil Deteksi: 🚫 Spam")
        else:
            st.success("Hasil Deteksi: ✅ Bukan Spam/Ham")
            
        st.write(f"Probabilitas Spam: {proba_spam:.2f}")
    else:
        st.warning("Silakan masukkan pesan SMS terlebih dahulu!")
