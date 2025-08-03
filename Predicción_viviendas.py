#Importamos nuestras herramientas para nuestra app web de predicción de precios de viviendas 

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model

# Título de la aplicación
st.title("Predicción de Precios de Viviendas")

# Subtítulo
st.markdown("""Esta aplicación predice el precio de una vivienda en función de su área, número de habitaciones y edad.""")

# Cargar los datos modelo
df = pd.read_csv("C:\\Users\\USUARIO\\OneDrive\\Documentos\\ciencia de datos\\proyectos de ciencia de datos\MLregresion\\Prediccion viviendas\\deptos.csv")

# Preprocesamiento de la columna bedrooms, para usarla en el modelo de regresión
df['bedroms'] = df['bedroms'].fillna(df['bedroms'].median())

# Entrenar el modelo
reg = linear_model.LinearRegression()
reg.fit(df.drop('price', axis='columns'), df['price'])

# Widgets para la entrada de datos del usuario
st.sidebar.header("Ingresa los detalles de la vivienda")
area = st.sidebar.number_input("Área (pies cuadrados)", min_value=0, value=3000)
bedrooms = st.sidebar.number_input("Número de habitaciones", min_value=0, value=3)
age = st.sidebar.number_input("Edad de la vivienda (años)", min_value=0, value=40)

# Botón para realizar la predicción
if st.sidebar.button("Predecir Precio"):
    # Realizar la predicción
    prediction = reg.predict([[area, bedrooms, age]])
    
    # Mostrar el resultado
    st.success(f"El precio predicho de la vivienda es: **${prediction[0]:,.2f}**")

# Mostrar los datos originales
st.subheader("Datos Originales")
st.write(df)

# Mostrar los coeficientes del modelo
st.subheader("Coeficientes del Modelo")
st.write(f"Coeficientes: {reg.coef_}")
st.write(f"Intercepto: {reg.intercept_}")