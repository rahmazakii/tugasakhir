import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image


# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model("my_model.h5")

# Panggil model di luar fungsi
model = load_trained_model()  # Panggil di luar fungsi agar model dimuat


# Define the classes (adjust based on your model's training labels)
classes = ["blight", "blast", "tungro"]  # Ganti dengan nama kelas sebenarnya

# Preprocess the image for prediction
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize sesuai input model Anda
    img_array = np.array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    return img_array


# Streamlit App
#st.title("Klasifikasi Penyakit padi")
#st.write("Aplikasi ini memprediksi penyakit padi beredasarkan gambar daun padi yang Anda unggah.")

st.markdown('<div class="title">Klasifikasi Penyakit Padi</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Aplikasi ini memprediksi penyakit padi berdasarkan gambar daun yang Anda unggah.</div>', unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("Unggah gambar daun", type=["png", "jpg", "jpeg"])
st.markdown('<div class="uploader"></div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True, output_format="auto")
    st.markdown('<div class="image-preview"></div>', unsafe_allow_html=True)

    # Preprocess and predict
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display the result
    st.markdown(f'<div class="result">Prediksi Penyakit: {predicted_class}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence">Kepercayaan: {confidence:.2f}%</div>', unsafe_allow_html=True)

    
    # Menambahkan CSS Styling
st.markdown("""
    <style>
    body {
        background-color: #f7f7f7; /* Warna latar belakang aplikasi */
        font-family: Arial, sans-serif; /* Font aplikasi */
    }
    .title {
        font-size: 32px;
        color: #2e7d32; /* Warna hijau untuk judul */
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .description {
        font-size: 16px;
        color: #555555;
        text-align: center;
        margin-bottom: 30px;
    }
    .uploader {
        text-align: center;
        margin-bottom: 30px;
    }
    .image-preview {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%; /* Ukuran gambar */
        border: 2px solid #ddd;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .result {
        font-size: 20px;
        color: #2e7d32; /* Warna hijau untuk hasil prediksi */
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .confidence {
        font-size: 18px;
        color: #555555; /* Warna abu-abu untuk tingkat kepercayaan */
        text-align: center;
        margin-top: 10px;
    }
    
    </style>
""", unsafe_allow_html=True)

