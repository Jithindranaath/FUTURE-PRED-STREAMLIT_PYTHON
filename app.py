import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open(r'C:\Users\ADMIN\OneDrive\Desktop\DS\2\3\linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Custom CSS for Styling
st.markdown("""
    <style>
        body {
            background-color: #D6EAF8;
        }
        .main {
            background-color: #D6EAF8;
            padding: 20px;
            border-radius: 10px;
        }
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #4CAF50;
        }
        .subtext {
            text-align: center;
            font-size: 18px;
            color: #555;
        }
        .prediction {
            font-size: 24px;
            color: #ff5733;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #888;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title with HTML Styling
st.markdown('<p class="title">💰 Salary Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Predict the salary based on years of experience</p>', unsafe_allow_html=True)

# Input Box for User
yr_exp = st.number_input('Enter Years of Experience:', min_value=0.0, max_value=50.0, value=1.0, step=0.5)

# Predict Button with Custom Styling
if st.button('💡 Predict Salary'):
    exp_input = np.array([[yr_exp]])  # Reshape input for model
    prediction = model.predict(exp_input)  # Predict salary

    # Display Result with Custom Styled Text
    st.markdown(f'<p class="prediction">Predicted Salary: ${prediction[0]:,.2f}</p>', unsafe_allow_html=True)

# Footer
st.markdown('<p class="footer">🚀 Built with ❤️ using Streamlit</p>', unsafe_allow_html=True)
