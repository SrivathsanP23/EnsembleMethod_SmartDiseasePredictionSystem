import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix , r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import streamlit as st
import altair as alt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

# Load the dataset
pdata = pd.read_csv('parkinsons.csv')

# Separate features and target
X = pdata.drop(columns=['name', 'status'], axis=1)
Y = pdata['status']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

# Apply SMOTE to the scaled training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scale, Y_train)

# Initialize models
# Create individual classifiers with best parameters and class weights
log_reg = LogisticRegression(C=0.1, class_weight='balanced')
svm = SVC(C=10, probability=True, class_weight='balanced')
rf = RandomForestClassifier(max_depth=10, n_estimators=100, class_weight='balanced')

# Create a Voting Classifier
voting_clf = VotingClassifier(estimators=[('lr', log_reg), ('svc', svm), ('rf', rf)], voting='soft',weights=[1,2,1])

# Train the model
voting_clf.fit(X_train_smote, y_train_smote)
#ense=pickle.load(open('smartdiseaseprediction_ensemble.sav','rb'))
# Function to predict and interpret results
def predict(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data=scaler.transform(input_data_reshaped)
    prediction = voting_clf.predict(std_data)
    return prediction[0]

# Function to calculate accuracy
def calculate_accuracy():
    Y_pred = voting_clf.predict(X_test_scale)
    acc = accuracy_score(Y_test, Y_pred)
    return acc
def train_acc():
    Y_pred = voting_clf.predict(X_train_smote)
    tracc=accuracy_score(y_train_smote,Y_pred)
    return tracc

def r_s():
    train_pred = voting_clf.predict(X_train_smote)
    r2=r2_score(y_train_smote,train_pred)
    return r2
# Streamlit interface
def main():
    st.title('Parkinson\'s Disease Prediction')

    # Sidebar with input fields
    st.sidebar.header('Input Parameters')
    input_data = []
    for feature in X.columns:
        min_val = X[feature].min()
        max_val = X[feature].max()
        val = st.sidebar.slider(f"{feature} ", min_value=float(min_val), max_value=float(max_val))
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
    st.write(f'Test Accuracy: {acc*100:.3f}%')
    tracc=train_acc()
    st.write(f'Train Accuracy:{tracc*100:.3f}%')
    r2=r_s()
    st.write(f'R2 score :{r2:.3f}')
    
    # Display distribution of target variable
    st.subheader('Distribution of Target Variable')
    st.write(pdata['status'].value_counts())
    st.bar_chart(pdata['status'].value_counts())

    # Display pair plot of selected features
    st.subheader('Pair Plot of Selected Features')
    columns_to_plot = ['MDVP:Fo(Hz)','MDVP:APQ', 'MDVP:Fhi(Hz)','spread1','spread2','MDVP:Shimmer(dB)',  'status']

    # Melting the dataframe for easier plotting with Altair
    melted_df = pd.melt(pdata[columns_to_plot], id_vars=['status'], value_vars=columns_to_plot[:-1], var_name='measurement', value_name='value')

    # Create a pairplot using Altair
    chart = alt.Chart(melted_df).mark_point().encode(
        x=alt.X('measurement:N', title='Measurement'),
        y=alt.Y('value:Q', title='Value'),
        color=alt.Color('status:N', legend=alt.Legend(title="Status")),
        tooltip=['measurement', 'value', 'status']
    ).facet(
        column='measurement:N',
    ).interactive()

    st.altair_chart(chart, use_container_width=True)
    test_pred = voting_clf.predict(X_test_scale)
    
    # Display confusion matrix
    st.subheader('Confusion Matrix')
    cm = confusion_matrix(Y_test,test_pred)
    st.write(cm)

if __name__ == '__main__':
    main()