

import streamlit as st
import joblib

# Configuración de la página
st.set_page_config(page_title="Amazon Sentiment AI", page_icon="📦")

st.title("Analizador de Sentimientos con IA")
st.markdown("Esta aplicación usa un modelo **LightGBM** entrenado con millones de reseñas de Amazon para clasificar comentarios.")

# Carga de modelos
@st.cache_resource # Esto evita que el modelo se recargue en cada clic, ahorrando RAM

base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'modelo_amazon.pkl')
vect_path = os.path.join(base_path, 'vectorizador.pkl')
    
model = joblib.load(model_path)
vectorizer = joblib.load(vect_path)

# Interfaz de usuario
review_input = st.text_area("Introduce la reseña en inglés:", placeholder="Example: This product is amazing, I love the...")

if st.button("Analizar Sentimiento"):
    if review_input.strip() != "":
        # 1. Limpieza básica
        clean_text = review_input.lower().replace(r"[^a-zA-Z\s]", "")
        
        # 2. Transformación
        vector = vectorizer.transform([clean_text])
        
        # 3. Predicción
        prediction = model.predict(vector)
        
        # 4. Mostrar resultado
        if prediction[0] == 2:
            st.success("✨ Sentimiento: POSITIVO")
            st.balloons()
        else:
            st.error("re: Sentimiento: NEGATIVO")
    else:
        st.warning("Por favor, escribe algo primero.")
