#making a web page and deploying our model

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("C:/Users/shris/Downloads/trained_model.sav", 'rb'))

input_data = [6,148,72,35,0,33.6,0.627,50]
# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)


#create a function for prediction
def diabetes_prediction(input_data):
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    if(prediction==1) :
        return "Diabeteic"
    else :
        return "Non Diabeteic"


def main():
    st.title('Diabetic Prediction Web Page')

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose amount')
    BloodPressure = st.text_input('Blood pressure')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction= st.text_input('DiabetesPedigreeFunction')
    Age = st.text_input('Age of the person')

    #code for prediction
    diagnosis = ''

    #creating a button for prediction

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)


if __name__ == '__main__':
    main()








