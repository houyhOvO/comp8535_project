import streamlit as st
import pickle
import numpy as np

# load the model
with open('./models/linear_regression.pkl', 'rb') as f:
    linear_model = pickle.load(f)
with open('./models/random_forest_regressor.pkl', 'rb') as f:
    forest_model = pickle.load(f)
with open('./models/decision_tree.pkl', 'rb') as f:
    decision_model = pickle.load(f)
with open('./models/rbf_svr.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('./models/xgboost.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('./models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
try:
    with open('./models/pca.pkl', 'rb') as pca_file:
        pca = pickle.load(pca_file)
except FileNotFoundError:
    pca = None

st.sidebar.title('**Select Your Configuration Here** 👇')
model_choice = st.sidebar.radio("Please select a model to use", ["Linear Regression", "Random Forest", "Decision Tree",
                                                                 "Support Vector Machine", "XGBoost"])

st.title("Welcome to the housing price prediction app!⚡")
with st.expander("What is this app about?"):
    st.write('''
    This app is designed to help you predict the price of housing.\n
    Just enter numbers and a few clicks, you'll get the price！
    ''')


if model_choice == "Linear Regression":
    # get user's input as features
    crim = st.number_input('CRIM: per capita crime rate by town')
    zn = st.number_input('ZN: proportion of residential land zoned')
    indus = st.number_input('INDUS: proportion of non-retail business acres per town')
    chas = st.number_input('CHAS:Charles River dummy variable(0/1)')
    nox = st.number_input('NOX:nitric oxides concentration')
    rm = st.number_input('RM: average number of rooms per dwelling')
    age = st.number_input('AGE: proportion of owner-occupied units built prior to 1940')
    dis = st.number_input('RIS: weighted distances to five Boston employment centres')
    rad = st.number_input('RAD: index of accessibility to radial highways')
    tax = st.number_input('TAX: full-value property-tax rate per $10,000')
    ptratio = st.number_input('PTRATIO: pupil-teacher ratio by town')
    b = st.number_input('B: the proportion of blacks by town')
    lstat = st.number_input('LSTAT: lower status of the population(%)')

    submit = st.button(label='Get Price')

    # calculate the result
    if submit:
        features = np.zeros((1, 13))
        feature_list = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]
        for i, feature in enumerate(feature_list):
            features[0, i] = feature

        # Normalized input feature
        features_std = scaler.transform(features)
        # Apply PCA
        if pca:
            features_std = pca.transform(features_std)
        # Make the prediction
        prediction = linear_model.predict(features_std)
        st.write(f"The predicted price is: ${prediction[0]:.2f}")


elif model_choice == "Random Forest":
    # st.write("Random Forest model interface will be here.")
    # get user's input as features
    crim = st.number_input('CRIM: per capita crime rate by town')
    zn = st.number_input('ZN: proportion of residential land zoned')
    indus = st.number_input('INDUS: proportion of non-retail business acres per town')
    chas = st.number_input('CHAS: Charles River dummy variable(0/1)')
    nox = st.number_input('NOX: nitric oxides concentration')
    rm = st.number_input('RM: average number of rooms per dwelling')
    age = st.number_input('AGE: proportion of owner-occupied units built prior to 1940')
    dis = st.number_input('RIS: weighted distances to five Boston employment centres')
    rad = st.number_input('RAD: index of accessibility to radial highways')
    tax = st.number_input('TAX: full-value property-tax rate per $10,000')
    ptratio = st.number_input('PTRATIO: pupil-teacher ratio by town')
    b = st.number_input('B: the proportion of blacks by town')
    lstat = st.number_input('LSTAT: lower status of the population(%)')

    submit = st.button(label='Get Price')

    # calculate the result
    if submit:
        features = np.zeros((1, 13))
        feature_list = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]
        for i, feature in enumerate(feature_list):
            features[0, i] = feature

        # Normalized input feature
        features_std = scaler.transform(features)
        # Apply PCA
        if pca:
            features_std = pca.transform(features_std)
        # Make the prediction
        prediction = forest_model.predict(features_std)
        st.write(f"The predicted price is: ${prediction[0]:.2f}")


elif model_choice == "Decision Tree":
    # st.write("Decision Tree model interface will be here.")
    # get user's input as features
    crim = st.number_input('CRIM: per capita crime rate by town')
    zn = st.number_input('ZN:proportion of residential land zoned')
    indus = st.number_input('INDUS: proportion of non-retail business acres per town')
    chas = st.number_input('CHAS:Charles River dummy variable(0/1)')
    nox = st.number_input('NOX:nitric oxides concentration')
    rm = st.number_input('RM: average number of rooms per dwelling')
    age = st.number_input('AGE: proportion of owner-occupied units built prior to 1940')
    dis = st.number_input('RIS: weighted distances to five Boston employment centres')
    rad = st.number_input('RAD: index of accessibility to radial highways')
    tax = st.number_input('TAX: full-value property-tax rate per $10,000')
    ptratio = st.number_input('PTRATIO: pupil-teacher ratio by town')
    b = st.number_input('B: the proportion of blacks by town')
    lstat = st.number_input('LSTAT: lower status of the population(%)')

    submit = st.button(label='Get Price')

    # calculate the result
    if submit:
        features = np.zeros((1, 13))
        feature_list = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]
        for i, feature in enumerate(feature_list):
            features[0, i] = feature

        # Normalized input feature
        features_std = scaler.transform(features)
        # Apply PCA
        if pca:
            features_std = pca.transform(features_std)
        # Make the prediction
        prediction = decision_model.predict(features_std)
        st.write(f"The predicted price is: ${prediction[0]:.2f}")
        
elif model_choice == "Support Vector Machine":
    # st.write("Decision Tree model interface will be here.")
    # get user's input as features
    crim = st.number_input('CRIM: per capita crime rate by town')
    zn = st.number_input('ZN:proportion of residential land zoned')
    indus = st.number_input('INDUS: proportion of non-retail business acres per town')
    chas = st.number_input('CHAS:Charles River dummy variable(0/1)')
    nox = st.number_input('NOX:nitric oxides concentration')
    rm = st.number_input('RM: average number of rooms per dwelling')
    age = st.number_input('AGE: proportion of owner-occupied units built prior to 1940')
    dis = st.number_input('RIS: weighted distances to five Boston employment centres')
    rad = st.number_input('RAD: index of accessibility to radial highways')
    tax = st.number_input('TAX: full-value property-tax rate per $10,000')
    ptratio = st.number_input('PTRATIO: pupil-teacher ratio by town')
    b = st.number_input('B: the proportion of blacks by town')
    lstat = st.number_input('LSTAT: lower status of the population(%)')

    submit = st.button(label='Get Price')

    # calculate the result
    if submit:
        features = np.zeros((1, 13))
        feature_list = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]
        for i, feature in enumerate(feature_list):
            features[0, i] = feature

        # Normalized input feature
        features_std = scaler.transform(features)
        # Apply PCA
        if pca:
            features_std = pca.transform(features_std)
        # Make the prediction
        prediction = svm_model.predict(features_std)
        st.write(f"The predicted price is: ${prediction[0]:.2f}")

elif model_choice == "XGBoost":
    # st.write("Decision Tree model interface will be here.")
    # get user's input as features
    crim = st.number_input('CRIM: per capita crime rate by town')
    zn = st.number_input('ZN:proportion of residential land zoned')
    indus = st.number_input('INDUS: proportion of non-retail business acres per town')
    chas = st.number_input('CHAS:Charles River dummy variable(0/1)')
    nox = st.number_input('NOX:nitric oxides concentration')
    rm = st.number_input('RM: average number of rooms per dwelling')
    age = st.number_input('AGE: proportion of owner-occupied units built prior to 1940')
    dis = st.number_input('RIS: weighted distances to five Boston employment centres')
    rad = st.number_input('RAD: index of accessibility to radial highways')
    tax = st.number_input('TAX: full-value property-tax rate per $10,000')
    ptratio = st.number_input('PTRATIO: pupil-teacher ratio by town')
    b = st.number_input('B: the proportion of blacks by town')
    lstat = st.number_input('LSTAT: lower status of the population(%)')

    submit = st.button(label='Get Price')

    # calculate the result
    if submit:
        features = np.zeros((1, 13))
        feature_list = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]
        for i, feature in enumerate(feature_list):
            features[0, i] = feature

        # Normalized input feature
        features_std = scaler.transform(features)
        # Apply PCA
        if pca:
            features_std = pca.transform(features_std)
        # Make the prediction
        # prediction = xgb_model.predict(features_std)
        # st.write(f"The predicted price is: ${prediction[0]:.2f}")

