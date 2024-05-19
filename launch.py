import streamlit as st
import pickle
import numpy as np

# load the model
with open('./models/linear_regression.pkl', 'rb') as f:
    model = pickle.load(f)

st.sidebar.title('**Select Your Configuration Here** 👇')
st.sidebar.radio("Please select a model to use", ["Linear Regression", "Random Forest", "Decision Tree"])

st.title("Welcome to the housing price prediction app!⚡")
with st.expander("What is this app about?"):
    st.write('''
    This app is designed to help you predict the price of housing.\n
    Just enter numbers and a few clicks, you'll get the price！
    ''')
# st.number_input('RM average number of rooms per dwelling')
# st.number_input('LSTAT lower status of the population(%)')
# st.number_input('PTRATIO  pupil-teacher ratio by town')
#
# submit = st.button(label='Get Price')

# 获取用户输入的特征值
crim = st.number_input('CRIM: per capita crime rate by town')
zn = st.number_input('ZN: proportion of residential land zoned for lots over 25,000 sq. ft.')
indus = st.number_input('INDUS: proportion of non-retail business acres per town')
chas = st.number_input('CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)')
nox = st.number_input('NOX: nitric oxides concentration (parts per 10 million)')
rm = st.number_input('RM: average number of rooms per dwelling')
age = st.number_input('AGE: proportion of owner-occupied units built prior to 1940')

submit = st.button(label='Get Price')

# 进行预测
features = np.array([[crim, zn, indus, chas, nox, rm, age]])
prediction = model.predict(features)

st.write(f"The predicted price is: ${prediction[0]:.2f}")