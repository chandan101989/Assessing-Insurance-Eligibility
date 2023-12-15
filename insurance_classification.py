import pandas as pd
import streamlit as st 
from pickle import load

st.title('Model: Assessing-Insurance-Eligibility')

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.number_input("AGE")
    data = {'age':age
            }
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

# load the model from disk
loaded_model = load(open('logistic_model.pkl', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')




