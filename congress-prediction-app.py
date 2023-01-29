import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Congress Prediction App

This app predicts the political party of congress reps based on input features such as religion, age, and tenure.

If you have a CSV file with other political rep information, feel free to upload it in the section to the left. 
The file must in the the format of: <Party,Age,Tenure,Religion> 

Limit of 200 MB per file.

Data obtained entirely from Wikipedia.
""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Religion = st.sidebar.selectbox('Religion', ('Adventist', 'Agnostic', 'Antiochian Orthodox', 'Baptist',
                                                     'Buddhist', 'Catholic', 'Congregationalist',
                                                     'Eastern Roman Catholic', 'Episcopalian', 'Greek Orthodox',
                                                     'Hindu', 'Islam', 'Jewish', 'Lutheran', 'Messianic Jewish',
                                                     'Methodist', 'Mormon', 'Non-Denominational Protestant',
                                                     'Nondenominational Protestant', 'Pentecostal', 'Presbyterian',
                                                     'Quaker', 'Reformed', 'Restorationist', 'Roman Catholic',
                                                     'Roman Roman Catholic', 'Russian Orthodox', 'Sunni Islam',
                                                     'Unitarian Universalist', 'Unknown',
                                                     'Unspecified Eastern Orthodox', 'Unspecified Protestant',
                                                     'Wesleyan-Holiness Evangelical'))
        Age = st.sidebar.slider('Age', 26, 89, 60)
        Tenure = st.sidebar.slider('Tenure', 0, 42, 6)
        data = {'Religion': Religion,
                'Age': Age,
                'Tenure': Tenure}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire congress dataset
# This will be useful for the encoding phase
congress_raw = pd.read_csv('congress_cleaned.csv')
congress = congress_raw.drop(columns=['Party'])
df = pd.concat([input_df, congress], axis=0)


encode = ['Religion']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]  # Selects only the first row (the user input data)

# Reads in saved classification model
load_clf = pickle.load(open('congress_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
congress_parties = np.array(['Democratic', 'Republican', 'Independent'])
st.write("Predicted Party: " + congress_parties[prediction][0])

st.subheader('Prediction Probability')
prediction_proba = prediction_proba[0]
st.write("Democratic : {}%".format(round(prediction_proba[0]*100,1)))
st.write("Republican : {}%".format(round(prediction_proba[1]*100,1)))
st.write("Independent : {}%".format(round(prediction_proba[2]*100,1)))

