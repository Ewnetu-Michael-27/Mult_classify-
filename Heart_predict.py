import streamlit as st 
import pandas as pd 


st.set_page_config(
    page_title="Prediction of CAD"
)

st.title("Applying Several Classification Models to Predict the Presence of CAD")

st.sidebar.success("Select a Page Above")

df=pd.read_csv("df_heart_clean_2.csv")
df_new=pd.read_csv("df_heart_clean_2.csv")



list_val=st.multiselect("Pick varaibles",
                              ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"] ,
                               ["age","sex","cp","thalach","exang","ca","thal"])

button=st.button("Try!")
if button:
    df_new=df[list_val]
    
button_2=st.button("Export")
if button_2:
    df_new.to_csv()
