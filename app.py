import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Memuat model yang disimpan dalam file .keras
model = tf.keras.models.load_model('model.keras')

# Fungsi untuk memproses gambar dan melakukan prediksi
def predict_image(image):
    image = image.resize((150, 150))
    # Konversi gambar menjadi array numpy dan normalisasi
    image = np.array(image) / 255.0
    # Tambahkan dimensi batch (1, 192, 192, 3)
    image = np.expand_dims(image, axis=0)
    # Lakukan prediksi
    predictions = model.predict(image)
    return predictions

# Antarmuka Pengguna dengan Streamlit
st.title("Scissors, Rock, Paper Image Prediction ✌️🪨📄")

# Unggah gambar
uploaded_image = st.file_uploader("Upload Hand Image", type=["jpg", "png", "jpeg"])

# Jika gambar diunggah, proses dan tampilkan prediksi
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    
    # Ubah ukuran gambar menggunakan PIL (dalam cm ke piksel)
    width_in_cm = 5  # Lebar yang diinginkan dalam cm
    dpi = 96  # Umumnya DPI standar untuk layar adalah 96
    width_in_pixels = int(width_in_cm * dpi / 2.54)  # Mengkonversi cm ke piksel
    
    # Resize gambar agar lebar sesuai dengan 7 cm
    image = image.resize((width_in_pixels, int(image.height * width_in_pixels / image.width)))
    
    # Menampilkan gambar setelah diubah ukurannya
    st.image(image, caption="Uploaded Image")
    
    # Prediksi gambar
    predictions = predict_image(image)
    labels = ['Rock', 'Scissors', 'Paper']
    predicted_class = labels[np.argmax(predictions)]

    # Tampilkan hasil prediksi
    st.write(f"Model Predicts: {predicted_class} 🎉")
    st.write(f"With Confidence Score: {np.max(predictions) * 100:.2f}%")