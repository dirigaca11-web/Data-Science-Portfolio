
import streamlit as st
import joblib

st.title("Analizador de Sentimientos - Amazon Reviews")
review = st.text_area("Escribe tu reseña aquí:")

model = joblib.load('modelo_amazon.pkl')
vectorizer = joblib.load('vectorizador.pkl')

if st.button("Predecir"):
    vector = vectorizer.transform([review.lower()])
    pred = model.predict(vector)
    res = "Positivo" if pred[0] == 2 else "Negativo"
    st.success(f"El sentimiento es: {res}")