import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import altair as alt 
import plotly.graph_objects as go 
import PIL
import plotly.express as px 
import statsmodels.api as sm

st.set_page_config(
    page_title="Prediction of CAD"
)

st.title("Applying Several Classification Models to Predict the Presence of CAD")
st.write("Coronary artery disease (CAD) is caused by plaque buildup in the wall of the arteries that supply blood to the heart. Accoring to CDC, it is the most common type of heart disease in the US, killing 375,476 people in 2021.")
st.write("The dataset that was generated to study discriminant function models for estimating probabilities of coronary artery disease from clinical and non-invasive test results of 303 patients from Cleveland Clinic in Cleveland, Ohio. Concretely, the dataset explores the relationship between 13 clinical and non-invasive tests and the presence of CAD for the goal of building a prediction model.")


st.sidebar.success("Select a Page Above")

st.text("")
st.markdown("***")

st.write("Below, P-values for each feature are shown. The points are red for those feature with P-value >0.05 (**See explanation below image**)")

df=pd.read_csv("df_heart_clean_2.csv")
col=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]

X=df[col]
y=df["num"]

logit_model=sm.Logit(y,X)
result_1=logit_model.fit()
p_values=[]
    
for i in result_1.pvalues:
    p_values.append(i)
    
col=X.columns 
df_p_val=pd.DataFrame(list(zip(p_values, col)), columns=["P_values", "Variables"])
chart_1 = alt.Chart(df_p_val).mark_point().encode(
    x='Variables:N',
    y="P_values:Q",
    # The highlight will be set on the result of a conditional statement
    color=alt.condition(
        alt.datum.P_values > 0.05,  
        alt.value('red'),     
        alt.value('blue')   
        ),
        tooltip=["Variables","P_values"]
    ).properties(width=600)

tab1, tab2, tab3=st.tabs(["P_values chart","Variables with P_value>0.05", "Variables with P_value<0.05"])
with tab1:
    st.altair_chart(chart_1,use_container_width=True)
with tab2:
    st.dataframe(df_p_val[df_p_val["P_values"]>0.05])
with tab3:
    st.dataframe(df_p_val[df_p_val["P_values"]<0.05])
with st.expander("See explanation"):
    st.write("In the previous page, IDA/EDA showed that 5 features have weak influence on the presence or absence of CAD (Chol, Trestbps, FBS, Restecg, Slope). The P-value analysis flagged those features to have p_value>0.05 ")












