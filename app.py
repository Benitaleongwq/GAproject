import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the trained model
model = load_model('best_credit_card_model')  # Change 'your_model_name' to the actual model file name without the .pkl extension

# Streamlit App Title
st.title('PyCaret Model Deployment with Streamlit')

# Input form for prediction
st.write("Input the features for prediction:")

# Mapping dictionaries for categorical data
q1_mapping = {
    '1': 'Always', 
    '2': 'Often', 
    '3': 'Seldom', 
    '4': 'Never'
}

q2_mapping = {
    '1': 'Groceries', 
    '2': 'Dining', 
    '3': 'Travel- Air Miles', 
    '4': 'Petrol',
    '5': 'Entertainment',
}

# Create a mapping dictionary
q3_mapping = {
    '1': '$30,000 to $49 999', 
    '2': '$50 000 to 74 999', 
    '3': '$75 000 to 99 999', 
    '4': '$100 000 to 149 999',
    '5': '>= $150K',
}

q4_mapping = {
    '1': 'Citibank', 
    '2': 'DBS', 
    '3': 'UOB', 
    '4': 'HSBC',
    '5': 'Standard Chart',
}

q5_mapping = {
    '1': 'Male', 
    '2': 'Female', 
}

q6_mapping = {
    '1': '1-3', 
    '2': '3-5', 
    '3': '5-7', 
    '4': '7+',
}

q7_mapping = {
    '1': 'High school diploma', 
    '2': 'Associate degree', 
    '3': 'Bachelor’s degree', 
    '4': 'Trade school certification',
    '5': 'Master\'s degree'
}

q8_mapping = {
    '1': 'Singaporean', 
    '2': 'Permanent Resident(PR)', 
    '3': 'Foreigner', 
}

q9_mapping = {
    '1': '21-29', 
    '2': '30-39', 
    '3': '40-49', 
    '4': '50-59',
    '5': '> 60'
}

q10_mapping = {
    '1': 'No min spendings required', 
    '2': 'Cashback and rewards', 
    '3': 'Miles', 
}

input_data = {
    'q1': st.selectbox('How often do you use your credit card?', list(q1_mapping.values())),
    'q2': st.selectbox('What is your most common spending category?', list(q2_mapping.values())),
    'q3': st.selectbox('What is your annual income??', list(q3_mapping.values())),
    'q4': st.selectbox('Which bank issued your credit card?', list(q4_mapping.values())),
    'q5': st.selectbox('What is your gender?', list(q5_mapping.values())),
    'q6': st.selectbox('How many dependents do you have?', list(q6_mapping.values())),
    'q7': st.selectbox('What is your highest level of education?', list(q7_mapping.values())),
    'q8': st.selectbox('What is your nationality?', list(q8_mapping.values())),
    'q9': st.selectbox('What is your age group?', list(q9_mapping.values())),
    'q10': st.selectbox('What do you value most in a credit card?', list(q10_mapping.values())),
}

# Convert the selected options back to corresponding keys (reverse mapping)
reverse_mappings = {
    'q1': {v: k for k, v in q1_mapping.items()},
    'q2': {v: k for k, v in q2_mapping.items()},
    'q3': {v: k for k, v in q3_mapping.items()},
    'q4': {v: k for k, v in q4_mapping.items()},
    'q5': {v: k for k, v in q5_mapping.items()},
    'q6': {v: k for k, v in q6_mapping.items()},
    'q7': {v: k for k, v in q7_mapping.items()},
    'q8': {v: k for k, v in q8_mapping.items()},
    'q9': {v: k for k, v in q9_mapping.items()},
    'q10': {v: k for k, v in q10_mapping.items()},
}

# Apply reverse mapping to convert selected values back to numeric codes
for key, value in input_data.items():
    input_data[key] = reverse_mappings[key][value]

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Make predictions when the user clicks the button
if st.button('Predict'):
    predictions = predict_model(model, data=input_df)
    st.write('Predictions:', predictions)
