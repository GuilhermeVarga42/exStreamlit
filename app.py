import streamlit as st
import pickle
import numpy as np
import pandas as pd

def load_model():
    model = pickle.load(open('trained_model.sav', 'rb'))
    return model

data = load_model()


st.title("Previsão de Portador de Diabetes")

st.write("Este aplicativo é uma demonstração de como um modelo de Machine Learning pode ser utilizado para prever se uma pessoa tem diabetes")

st.subheader("Por favor, preencha as informações abaixo:")

def load_data():
        df = pd.read_csv('diabetes.csv')
        df = df[['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]

        return df

df = load_data()


gravidez = st.slider("Número de vezes grávida", int(df['Pregnancies'].min()), int(df['Pregnancies'].max()), 1)
glicose = st.slider("Nível de glicose", int(df['Glucose'].min()), int(df['Glucose'].max()), 1)
pressao = st.slider("Pressão sanguínea", int(df['BloodPressure'].min()), int(df['BloodPressure'].max()), 1)
pele = st.slider("Espessura da Pele", int(df['SkinThickness'].min()), int(df['SkinThickness'].max()), 1)
insulina = st.slider("Nível de insulina", int(df['Insulin'].min()), int(df['Insulin'].max()), 1)
imc = st.slider("Índice de massa corporal", int(df['BMI'].min()), int(df['BMI'].max()), 1)
pedigree = st.slider("Função Pedigree Diabetes", df['DiabetesPedigreeFunction'].min(), df['DiabetesPedigreeFunction'].max(), 0.01)
idade = st.slider("Idade", int(df['Age'].min()), int(df['Age'].max()), 1)


button = st.button("Prever Diabetes")

if button:
        input = np.array([[gravidez, glicose, pressao, pele, insulina, imc, pedigree, idade]])
        input = input.astype(float)

        prediction = data.predict(input)
        
        st.title(f"O diagnóstico previsto é: {'Tem diabetes' if prediction[0] else 'Não tem diabetes'}.")