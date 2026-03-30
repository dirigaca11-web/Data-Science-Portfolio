
import streamlit as st
import joblib
import os

st.title("Analizador de Sentimientos - Amazon Reviews")
review = st.text_area("Escribe tu reseña aquí:")
base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, 'modelo_amazon.pkl')
vect_path = os.path.join(base_path, 'vectorizador.pkl')
model = joblib.load(model_path)
vectorizer = joblib.load(vect_path)

if st.button("Predecir"):
    vector = vectorizer.transform([review.lower()])
    pred = model.predict(vector)
    res = "Positivo" if pred[0] == 2 else "Negativo"
    st.success(f"El sentimiento es: {res}")
