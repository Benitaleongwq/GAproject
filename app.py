# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set Page configuration
# Read more at https://docs.streamlit.io/1.6.0/library/api-reference/utilities/st.set_page_config
st.set_page_config(page_title='Credit Card Recommendation', page_icon='🪪', layout='wide', initial_sidebar_state='expanded')


# Set title of the app
st.title('Credit Card')

# Load data
df = pd.read_csv('reviews.csv')


# Define options for the selectboxes
options1 = ["Always", "Often", "Seldom", "Never"]
options2 = ["Groceries", "Dining", "Travel", "Petrol", "Entertainment"]
options3 = ["$30,000 to $49,999", "$50,000 to $74 999", "$75,000 to $99,999", "$100,000 to $149,999", ">= $150K"]

# Define options for the selectboxes
options1 = ["Always", "Often", "Seldom", "Never"]
options2 = ["Groceries", "Dining", "Travel", "Petrol", "Entertainment"]
options3 = ["$30,000 to $49,999", "$50,000 to $74 999", "$75,000 to $99,999", "$100,000 to $149,999", ">= $150K"]
options4 = ["Always", "Often", "Seldom", "Never"]


# Create the first selectbox with a unique key
option1 = st.selectbox(
    'Do you have savings account in with any banks?',
    options1,
    key='selectbox_1'
)

# Create the second selectbox with a different unique key
option2 = st.selectbox(
    'What is the purpose for my credit card?',
    options2,
    key='selectbox_2'
)

# Create the third selectbox with another unique key
option3 = st.selectbox(
    'What is your annual income? ',
    options3,
    key='selectbox_3'
)


# Create the fourth selectbox with another unique key
option4 = st.selectbox(
    'Do you have savings account in with any banks?',
    options4,
    key='selectbox_4'
)

# Create the fifth selectbox with another unique key
option5 = st.selectbox(
    'What is your gender?',
    options5,
    key='selectbox_5'
)

# Create the sixth selectbox with another unique key
option6 = st.selectbox(
    'Do you have any dependents?',
    options6,
    key='selectbox_6'
)

# Create the seventh selectbox with another unique key
option7 = st.selectbox(
    'What is your academic background?',
    options7,
    key='selectbox_7'
)
# Create the eighth selectbox with another unique key
option8 = st.selectbox(
    'What is your nationality?',
    options8,
    key='selectbox_8'
)
# Create the ninth selectbox with another unique key
option9 = st.selectbox(
    'What is your age?',
    options9,
    key='selectbox_9'
)


# Display the selected options
st.write('You selected for the first selectbox:', option1)
st.write('You selected for the second selectbox:', option2)
st.write('You selected for the third selectbox:', option3)
st.write('You selected for the fourth selectbox:', option4)
st.write('You selected for the fifth selectbox:', option5)
st.write('You selected for the sixth selectbox:', option6)
st.write('You selected for the seventh selectbox:', option7)
st.write('You selected for the eighth selectbox:', option8)
st.write('You selected for the nineth selectbox:', option9)

# Set input widgets
st.sidebar.subheader('Select Criteria')
Q1 = st.sidebar.slider('What is your annual income?',   min_value=30000,
    max_value=150000,
    value=75000,
    step=5000)
Q2 = st.sidebar.slider('What is your age?',   min_value=18,
    max_value=69,
    value=69,
    step=1)


# Separate to X and y
X = df.drop('credit card', axis=1)
y = df['creditcard']
 

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = RandomForestClassifier(max_depth=2, max_features=4, n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Generate prediction based on user selected attributes
y_pred = model.predict([[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]])


 
 


# Display EDA
st.subheader('Exploratory Data Analysis')
st.write('The data is grouped by the class and the variable mean is computed for each class.')
groupby_species_mean = df.groupby('credit card').mean()
st.write(groupby_species_mean)
st.bar_chart(groupby_species_mean.T)

# Print input features
st.subheader('Variables in Data Set')
input_feature = pd.DataFrame([[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]],
                            columns=['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10'])
st.write(input_feature)

# Print Recommended Credit Card
st.subheader('Recommendation')
st.metric('Recommended Credit Card is :', y_pred[0], '')