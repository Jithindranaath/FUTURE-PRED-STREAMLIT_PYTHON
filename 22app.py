import streamlit as st
import pickle
import numpy as np

with open(r'C:\Users\ADMIN\OneDrive\Desktop\DS\2\3\linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
st.title('Salary Prediction App')
st.write('This app predicts the salary of an employee based on years of experience using simple linear regression.')
yr_exp=st.number_input('Enter yrs of exps:',min_value=0.0,max_value=50.0,value=1.0,step=0.5)
if st.button('predict salary'):
    exp_input=np.array([[yr_exp]])
    
    prediction=model.predict(exp_input)
    
        # Display the result
    st.success(f"The predicted salary for {yr_exp} years of experience is: ${prediction[0]:,.2f}")
    
    st.write('the model was trained using a dataset of salaries and yrs of exp.built model by me')
    #streamlit run 22app.py