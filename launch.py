import streamlit as st
import pickle
import numpy as np

# Load all models and scaler, PCA optionally
models = {
    "Linear Regression": "linear_regression.pkl",
    "Random Forest": "random_forest_regressor.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Support Vector Machine": "rbf_svr.pkl",
    "XGBoost": "xgboost.pkl"
}

# Load models
loaded_models = {}
for model_name, file_name in models.items():
    with open(f'./models/{file_name}', 'rb') as f:
        loaded_models[model_name] = pickle.load(f)

with open('./models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

try:
    with open('./models/pca.pkl', 'rb') as f:
        pca = pickle.load(f)
except FileNotFoundError:
    pca = None

st.set_page_config(layout="wide")
# Setup sidebar and main page
st.sidebar.title('**Select Your Configuration Here** 👇')
model_choice = st.sidebar.radio("Please select a model to use", list(models.keys()))

st.title("Welcome to the housing price prediction app!⚡")
with st.expander("What is this app about?"):
    st.write('''
    This app is designed to help you predict the price of housing.
    Just enter numbers and a few clicks, you'll get the price！
    ''')


# Function to collect user input
def get_user_input():
    # Create two columns for input features
    col1, col2 = st.columns(2)

    with col1:
        crim = st.number_input('CRIM: per capita crime rate by town', min_value=0.00)
        zn = st.number_input('ZN: proportion of residential land zoned', min_value=0.00)
        indus = st.number_input('INDUS: proportion of non-retail business acres per town', min_value=0.00)
        chas = st.radio('CHAS: Charles River dummy variable (0/1)', [0, 1])
        nox = st.number_input('NOX: nitric oxides concentration', min_value=0.00)
        rm = st.number_input('RM: average number of rooms per dwelling', min_value=0.00)

    with col2:
        age = st.number_input('AGE: proportion of owner-occupied units built prior to 1940', min_value=0.00)
        dis = st.number_input('DIS: weighted distances to five Boston employment centres', min_value=0.00)
        rad = st.number_input('RAD: index of accessibility to radial highways', min_value=0, step=1, format="%d")
        tax = st.number_input('TAX: full-value property-tax rate per $10,000', min_value=0.00)
        ptratio = st.number_input('PTRATIO: pupil-teacher ratio by town', min_value=0.00)
        b = st.number_input('B: proportion of blacks by town', min_value=0.00)
        lstat = st.number_input('LSTAT: lower status of the population(%)', min_value=0.00)

    features = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
    return features


# Make prediction based on model
def make_prediction(model, features):
    features_std = scaler.transform(features)
    if pca:
        features_std = pca.transform(features_std)
    prediction = model.predict(features_std)
    return prediction


features = get_user_input()
# Main interactive script
if st.button('Get Price'):
    model = loaded_models[model_choice]
    prediction = make_prediction(model, features)
    with st.container():
        st.write("## Prediction Result 🏠")
        st.markdown(f"### **Predicted Price:** `{prediction[0]:,.2f}` (in $1000's)")
        st.caption("This is an estimation of the house price based on the provided features.")