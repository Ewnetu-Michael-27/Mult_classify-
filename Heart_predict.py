import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import math 
import altair as alt 
import plotly.graph_objects as go 
from PIL import Image
import plotly.express as px 
import plotly.figure_factory as ff
import statsmodels.api as sm


st.set_page_config(
    page_title="Prediction of CAD"
)

st.title("Applying Several Classification Models to Predict the Presence of CAD")
st.write("Coronary artery disease (CAD) is caused by plaque buildup in the wall of the arteries that supply blood to the heart. Accoring to CDC, it is the most common type of heart disease in the US, killing 375,476 people in 2021.")
st.write("The dataset that was generated to study discriminant function models for estimating probabilities of coronary artery disease from clinical and non-invasive test results of 303 patients from Cleveland Clinic in Cleveland, Ohio. Concretely, the dataset explores the relationship between 13 clinical and non-invasive tests and the presence of CAD for the goal of building a prediction model.")


st.sidebar.success("Select from the options of 7 Pages")




st.text("")
st.markdown("***")
st.text("")

st.write("**Breif Feature Analysis**")
st.markdown('''
Feature selection is the process of choosing a subset of relevant features from a 
larger set of features to improve the performance of a model or reduce its complexity. 
Even though it has limitations, p-value analysis can be quickly used to study the feature's importance in the predictive model. 
''')

st.text("")

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
    st.write("IDA/EDA process performed [here](https://exploring-prediction-of-cad.streamlit.app/) showed that 5 features have weak influence on the presence or absence of CAD (Chol, Trestbps, FBS, Restecg, Slope). The P-value analysis flagged those features to have p_value>0.05 ")

st.text("")
st.markdown("***")
st.text("")

st.write("**Additional Preprocessing of Data**")
st.write("Before each model training, Standard Scalar method is used to regularize the data. Then, the imbalnce of the data is fixed.")


#Finding continous and categorical variables
def choose_cont(x, st):
        list_cont=[]
        if st=="cont":
            for i in x:
                if i in ["age","trestbps","chol","thalach","oldpeak"]:
                    list_cont.append(i)
        else:
            for i in x:
                if i in ["sex","cp","fbs","restecg","exang","slope","ca","thal"]:
                    list_cont.append(i)


        return list_cont


list_features=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
cont=choose_cont(list_features, "cont")
cat=choose_cont(list_features, "cat")
output=["num"]


X_cont=df[cont]
y=df[output].astype("category").to_numpy()

col=cont+["num"]
for_fig=pd.DataFrame((np.concatenate((X_cont, y), axis=1)), columns=col)
x_str=col[0] 
y_str=col[1]

st.text("")
st.markdown("***")
st.text("")

st.write("What the data looks like before the aditional preprocessing of data")
tab_1, tab_2=st.tabs(["Before Standard Scalr Process", "Before Balancing the Data"])


fig_1=alt.Chart(for_fig).mark_circle().encode(
        x=x_str,
        y=y_str, 
        color="num", 
        tooltip=[x_str, y_str, "num"]
        ).interactive()

labels=["0 (Absence)", "1 (Presence)"]
val=np.unique(y, return_counts=True)
values=[val[1][0], val[1][1]]

fig_2=go.Figure(data=[go.Pie(labels=labels, values=values)])



with tab_1:
    st.altair_chart(fig_1, use_container_width=True)
with tab_2:
    st.plotly_chart(fig_2)


st.text("")
st.markdown("***")
st.text("")

st.write("What the data looks like after the aditional preprocessing of data")

X_cont=df[cont].to_numpy()
my_scaler = StandardScaler()
my_scaler.fit(X_cont)
X_cont_sc=my_scaler.transform(X_cont)

X_cat=df[cat].to_numpy()
X_new=np.hstack((X_cont_sc, X_cat))

X_train,X_test,y_train,y_test = train_test_split(X_new,y,test_size=0.2,random_state=42)


samp = {0: len(y_train[y_train == 0]), 1: len(y_train[y_train == 0])}
sampler=RandomOverSampler(sampling_strategy=samp, random_state=42)
X_res, y_res=sampler.fit_resample(X_train, y_train)

tab_3, tab_4=st.tabs(["After Standard Scalr Process", "After Balancing the Data by Sampling"])


col=cont+["num"]
for_fig=pd.DataFrame((np.concatenate((X_cont_sc, y), axis=1)), columns=col)

fig_3=alt.Chart(for_fig).mark_circle().encode(
        x=x_str,
        y=y_str, 
        color="num", 
        tooltip=[x_str, y_str, "num"]
        ).interactive()
    
labels=["0 (Absence)", "1 (Presence)"]
val=np.unique(y_res, return_counts=True)
values=[val[1][0], val[1][1]]

fig_4=go.Figure(data=[go.Pie(labels=labels, values=values)])

with tab_3:
    st.altair_chart(fig_3, use_container_width=True)
with tab_4:
    st.plotly_chart(fig_4)













