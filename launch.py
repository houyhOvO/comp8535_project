import streamlit as st
import pickle
import numpy as np

# load the model
with open('./models/linear_regression.pkl', 'rb') as f:
    model = pickle.load(f)
with open('./models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
try:
    with open('./models/pca.pkl', 'rb') as pca_file:
        pca = pickle.load(pca_file)
except FileNotFoundError:
    pca = None

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
rm = st.number_input('RM: average number of rooms per dwelling')
lstat = st.number_input('LSTAT: lower status of the population(%)')
ptratio = st.number_input('PTRATIO: pupil-teacher ratio by town')

submit = st.button(label='Get Price')

# 输出预测结果
if submit:
    # 检查所有输入是否有效
    if rm > 0 and lstat > 0 and ptratio > 0:
        # 创建一个包含所有特征的数组，用 0 填充未使用的特征
        features = np.zeros((1, 13))
        features[0, 5] = rm
        features[0, 12] = lstat
        features[0, 10] = ptratio
        # 标准化输入特征
        features_std = scaler.transform(features)
        # 如果有PCA，应用PCA变换
        if pca:
            features_std = pca.transform(features_std)
        # 进行预测
        prediction = model.predict(features_std)
        st.write(f"The predicted price is: ${prediction[0]:.2f}")
    else:
        st.write("Please enter valid values for all features.")
