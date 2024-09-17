import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = io.imread(uploaded_file)

    if len(image.shape) == 3:
        image = image[:, :, 0]  # Преобразуем в черно-белое изображение, взяв один цветовой канал

    st.image(image, caption='Исходное изображение', use_column_width=True)

    k = st.slider('Выбери число сингулярных значений', 1, min(image.shape), value=50)
    U, sing_values, V = np.linalg.svd(image)
    U_k = U[:, :k]
    sigma_k = np.diag(sing_values[:k])
    V_k = V[:k, :]
    recon_img = np.dot(U_k, np.dot(sigma_k, V_k))
    
    recon_img = recon_img - np.min(recon_img)
    recon_img = recon_img / np.max(recon_img)

    st.image(recon_img, caption=f'Восстановленное изображение с {k} сингулярными значениями', use_column_width=True)

