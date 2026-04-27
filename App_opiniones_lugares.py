import base64

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))



st.set_page_config(page_title="Minería de opiniones", layout="wide", initial_sidebar_state="expanded")

# CARGA DE ARCHIVOS
@st.cache_data
def cargar_tablas():
    tabla_ciudades = pd.read_csv("tabla_resumen_ciudades.csv")
    tabla_lugares = pd.read_csv("tabla_resumen_lugares_filtrada.csv")
    return tabla_ciudades, tabla_lugares

@st.cache_resource
def cargar_modelo():
    vectorizador = joblib.load("vectorizador_tfidf.pkl")
    modelo = joblib.load("modelo_sentimiento.pkl")
    return vectorizador, modelo


tabla_ciudades, tabla_lugares = cargar_tablas()
vectorizador, modelo = cargar_modelo()

# FUNCION DE PREDICCION

def preprocesar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-zA-Z\\s]", " ", texto)
    tokens = word_tokenize(texto)
    tokens = [palabra for palabra in tokens if palabra not in stop_words]
    texto_limpio = " ".join(tokens)
    return texto_limpio


def predecir_sentimiento(texto: str):
    texto_procesado = preprocesar_texto(texto)
    texto_vectorizado = vectorizador.transform([texto_procesado])
    prediccion = modelo.predict(texto_vectorizado)[0]

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(texto_vectorizado)[0]
        clases = modelo.classes_
        probs_df = pd.DataFrame({
            "Clase": clases,
            "Probabilidad": probabilidades
        }).sort_values(by="Probabilidad", ascending=False)

        return prediccion, probs_df, texto_procesado

    return prediccion, None, texto_procesado


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()



#SIDEBAR
st.sidebar.title("Navegación")
opcion = st.sidebar.radio(
    "Ir a:",
    [
        
        "Análisis por ciudad",
        "Análisis por lugar",
        "Predicción de nueva reseña"
    ]
)



# PAGINA ANALISIS POR CIUDAD
if opcion == "Análisis por ciudad":
    st.title("Análisis por ciudad")

    ciudad = st.selectbox("Selecciona una ciudad", sorted(tabla_ciudades["City"].unique()))
    fila_ciudad = tabla_ciudades[tabla_ciudades["City"] == ciudad].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de reseñas", int(fila_ciudad["total_reviews"]))
    col2.metric("Total de lugares", int(fila_ciudad["total_places"]))
    col3.metric("Rating promedio", f"{fila_ciudad['rating_promedio']:.2f}")
    col4.metric("Índice de percepción", f"{fila_ciudad['indice_percepcion']:.2f}")

    st.subheader("Distribución de sentimientos en la ciudad")
    sentimientos_ciudad = {
        "Negativo": fila_ciudad["negativo"],
        "Neutral": fila_ciudad["neutral"],
        "Positivo": fila_ciudad["positivo"]
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sentimientos_ciudad.keys(), sentimientos_ciudad.values())
    ax.set_xlabel("Sentimiento")
    ax.set_ylabel("Cantidad")
    ax.set_title(f"Sentimientos en {ciudad}")
    st.pyplot(fig, transparent=True)

    st.subheader("Lugares de la ciudad")
    lugares_ciudad = tabla_lugares[tabla_lugares["City"] == ciudad].sort_values(
        by="indice_percepcion", ascending=False
    )
    st.dataframe(lugares_ciudad, width="stretch")

    st.subheader("Top lugares de la ciudad")
    top_ciudad = lugares_ciudad.head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_ciudad["Place"], top_ciudad["indice_percepcion"])
    ax.invert_yaxis()
    ax.set_xlabel("Índice de percepción")
    ax.set_ylabel("Lugar")
    ax.set_title(f"Top lugares en {ciudad}")
    plt.tight_layout()
    st.pyplot(fig, transparent=True)


# PAGINA 3: ANALISIS POR LUGAR
elif opcion == "Análisis por lugar":
    st.title("Análisis por lugar")

    ciudad_filtro = st.selectbox("Filtrar por ciudad", ["Todas"] + sorted(tabla_lugares["City"].dropna().unique().tolist()))

    if ciudad_filtro == "Todas":
        lugares_filtrados = tabla_lugares.copy()
    else:
        lugares_filtrados = tabla_lugares[tabla_lugares["City"] == ciudad_filtro].copy()

    lugar = st.selectbox("Selecciona un lugar", sorted(lugares_filtrados["Place"].unique()))
    fila_lugar = lugares_filtrados[lugares_filtrados["Place"] == lugar].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ciudad", fila_lugar["City"])
    col2.metric("Total de reseñas", int(fila_lugar["total_reviews"]))
    col3.metric("Rating promedio", f"{fila_lugar['rating_promedio']:.2f}")
    col4.metric("Índice de percepción", f"{fila_lugar['indice_percepcion']:.2f}")

    st.subheader("Proporción de sentimientos del lugar")
    proporciones = pd.DataFrame({
        "Sentimiento": ["Negativo", "Neutral", "Positivo"],
        "Proporción": [
            fila_lugar["prop_negativo"],
            fila_lugar["prop_neutral"],
            fila_lugar["prop_positivo"]
        ]
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(proporciones["Sentimiento"], proporciones["Proporción"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sentimiento")
    ax.set_ylabel("Proporción")
    ax.set_title(f"Distribución de sentimientos en {lugar}")
    st.pyplot(fig, transparent=True)

    st.subheader("Detalles del lugar")
    st.dataframe(pd.DataFrame([fila_lugar]),width="stretch")

# PAGINA 4: PREDICCION NUEVA
elif opcion == "Predicción de nueva reseña":
    st.title("Predicción de nueva reseña")
    st.write("Escribe una reseña y el modelo predecirá si es negativa, neutral o positiva.")

    texto_usuario = st.text_area(
        "Escribe aquí tu reseña",
        placeholder="Ejemplo: beautiful place clean peaceful excellent service"
    )

    if st.button("Predecir sentimiento"):
        if texto_usuario.strip() == "":
            st.warning("Por favor, escribe una reseña antes de predecir.")
        else:
            #prediccion, probs_df = predecir_sentimiento(texto_usuario)
            prediccion, probs_df, texto_procesado = predecir_sentimiento(texto_usuario)
            st.write("**Texto procesado:**", texto_procesado)

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if prediccion == "positivo":
                    st.markdown(
                        """
                        <div style="text-align: center;">
                            <img src="data:image/png;base64,{}" width="180">
                            <h3>Buena</h3>
                        </div>
                        """.format(get_base64_image("feliz.png")),
                        unsafe_allow_html=True
                        )

                elif prediccion == "neutral":
                    st.markdown(
                        """
                        <div style="text-align: center;">
                            <img src="data:image/png;base64,{}" width="180">
                            <h3>Regular</h3>
                        </div>
                        """.format(get_base64_image("neutra.png")),
                        unsafe_allow_html=True
                    )

                elif prediccion == "negativo":
                    st.markdown(
                        """
                        <div style="text-align: center;">
                            <img src="data:image/png;base64,{}" width="180">
                            <h3>Mala</h3>
                            
                        </div>
                        """.format(get_base64_image("mala.png")),
                        unsafe_allow_html=True
                    )

            #st.write(f"**Sentimiento predicho:** {prediccion}")


            if probs_df is not None:

                probs_plot = probs_df.copy()

                probs_plot["Clase"] = probs_plot["Clase"].replace({
                    "positivo": "Bueno",
                    "neutral": "Regular",
                    "negativo": "Malo"
                })

                colores = {
                    "Bueno": "#4CAF50",   # verde
                    "Regular": "#FFC107", # amarillo
                    "Malo": "#F44336"     # rojo
                }

                colores_pastel = [colores[clase] for clase in probs_plot["Clase"]]

                fig, ax = plt.subplots(figsize=(2, 2))


                fig.patch.set_alpha(0)
                ax.set_facecolor("none")

                wedges, texts, autotexts = ax.pie(
                    probs_plot["Probabilidad"],
                    labels=probs_plot["Clase"],
                    autopct="%1.2f%%",
                    startangle=90,
                    colors=colores_pastel
                )

                ax.set_title("Distribución de probabilidades", color="white", fontsize=5)
                ax.axis("equal")

                for text in texts:
                    #texto de clase
                    text.set_fontsize(5)

                for autotext in autotexts:
                    #texto de porcentaje
                    text.set_fontsize(4)

                plt.tight_layout()
                st.pyplot(fig, transparent=True)
                
                