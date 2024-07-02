import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pickle
import streamlit as st
# import seaborn as sns
# import matplotlib.pyplot as plt

# Load the dataset
pdata = pd.read_csv('parkinsons1.csv')

# Separate features and target
X = pdata.drop(columns=['name', 'status'], axis=1)
Y = pdata['status']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

# Initialize models
log_reg = LogisticRegression(max_iter=10000)
svm = SVC(probability=True)
rf = RandomForestClassifier()

# Create a Voting Classifier
voting_clf = VotingClassifier(estimators=[('lr', log_reg), ('svc', svm), ('rf', rf)], voting='soft')

# Train the model
voting_clf.fit(X_train, Y_train)

# Function to predict and interpret results
def predict(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = voting_clf.predict(input_data_reshaped)
    return prediction[0]

# Function to calculate accuracy
def calculate_accuracy():
    Y_pred = voting_clf.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    return acc

# Streamlit interface
def main():
    st.title('Parkinson\'s Disease Prediction')

    # Sidebar with input fields
    st.sidebar.header('Input Parameters')
    input_data = []
    for feature in X.columns:
        min_val = X[feature].min()
        max_val = X[feature].max()
        val = st.sidebar.slider(f"{feature} ", min_value=min_val, max_value=max_val)
        input_data.append(val)

    # Make prediction
    if st.sidebar.button('Predict'):
        prediction = predict(input_data)
        if prediction == 0:
            st.success('Prediction: The person does not have Parkinson\'s Disease')
        else:
            st.error('Prediction: The person has Parkinson\'s Disease')

    # Display accuracy
    st.subheader('Accuracy of Ensemble Model')
    acc = calculate_accuracy()
    st.write(f'Accuracy: {acc:.2f}')

    # Display distribution of target variable
    st.subheader('Distribution of Target Variable')
    st.write(pdata['status'].value_counts())
    st.bar_chart(pdata['status'].value_counts())

    # Display pair plot of selected features
    st.subheader('Pair Plot of Selected Features')
    columns_to_plot = ['MDVP-Jitter(%)',	'MDVP-Jitter(Abs)',	'MDVP-RAP',	'MDVP-PPQ',	'Jitter-DDP',	'MDVP-Shimmer', 'status']  # Replace with your columns
    sns.pairplot(pdata[columns_to_plot], hue='status', diag_kind='kde')
    
    

    # Display confusion matrix
    st.subheader('Confusion Matrix')
    cm = confusion_matrix(Y_test, voting_clf.predict(X_test))
    st.write(cm)

if __name__ == '__main__':
    main()
