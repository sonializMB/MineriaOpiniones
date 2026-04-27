# Minería de opiniones en reseñas turísticas

Este proyecto analiza reseñas turísticas para identificar la percepción de los visitantes sobre distintos lugares y ciudades. Para ello, se aplican técnicas de minería de texto, representación TF-IDF y modelos de clasificación de sentimientos.

## Objetivo

Identificar qué lugares turísticos presentan una mejor percepción entre los visitantes a partir del análisis de reseñas, clasificándolas en tres categorías:

- **positivo**
- **neutral**
- **negativo**

## Dataset

El conjunto de datos incluye, entre otras, las siguientes columnas:

- `City`: ciudad
- `Place`: lugar turístico
- `Review`: texto preprocesado en palabras clave
- `Raw_Review`: reseña original
- `Rating`: calificación otorgada por el usuario

## Metodología

El flujo general del proyecto fue el siguiente:

1. Carga e inspección del dataset
2. Limpieza y preparación de datos
3. Construcción de la variable de sentimiento a partir de `Rating`
4. División en entrenamiento y prueba
5. Representación del texto mediante **TF-IDF**
6. Entrenamiento de modelos de clasificación:
   - Naive Bayes
   - Regresión Logística
   - LinearSVC
7. Evaluación de desempeño
8. Selección del modelo final
9. Generación de tablas resumen por ciudad y lugar
10. Desarrollo de una aplicación visual en **Streamlit**



## Archivos principales

- `modelo_sentimiento.pkl` → modelo entrenado
- `vectorizador_tfidf.pkl` → vectorizador TF-IDF
- `tabla_resumen_ciudades.csv` → resumen por ciudad
- `tabla_resumen_lugares_filtrada.csv` → resumen por lugar
- `App_opiniones_lugares.py` → aplicación en Streamlit
- notebooks `.ipynb` → desarrollo del análisis y entrenamiento

## Dashboard

La aplicación desarrollada en Streamlit permite:


- explorar resultados por ciudad
- explorar resultados por lugar
- consultar percepción turística
- ingresar una nueva reseña y predecir su sentimiento
- mostrar probabilidades estimadas del modelo
