import streamlit as st

st.sidebar.title('**Select Your Configuration Here** 👇')
st.sidebar.radio("Please select a model to use", ["Linear Regression", "Random Forest", "Decision Tree"])

st.title("Welcome to the housing price prediction app!⚡")
with st.expander("What is this app about?"):
    st.write('''
    This app is designed to help you predict the price of housing.\n
    Just enter numbers and a few clicks, you'll get the price！
    ''')
st.number_input('RM average number of rooms per dwelling')
st.number_input('LSTAT lower status of the population(%)')
st.number_input('PTRATIO  pupil-teacher ratio by town')

submit = st.button(label='Get Price')
