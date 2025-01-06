import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import load_iris

model = joblib.load('./model.joblib')

st.title('Iris Species Prediction')
st.write('Predicting the species of Iris using Random Forest Model')

st.sidebar.header('Input Features')
sl = st.sidebar.slider('Sepal Length (cm):', 0.0, 10.0, 5.8)
sw = st.sidebar.slider('Sepal Width (cm):', 0.0, 10.0, 2.5)
pl = st.sidebar.slider('Petal Length (cm):', 0.0, 10.0, 6.0)
pw = st.sidebar.slider('Petal width (cm):', 0.0, 10.0, 4.0)

input_data = pd.DataFrame({'sepal length (cm)': [sl],
                            'sepal width (cm)': [sw],
                            'petal length (cm)': [pl],
                            'petal width (cm)': [pw]})

tar = load_iris().target_names

st.write("Input Data:")
st.write(input_data)

if st.button('Predict'):
    prediction = model.predict(input_data)
    st.success(f'The predicted class is {tar[prediction][0]}')
    