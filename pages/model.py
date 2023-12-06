import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import math 
import altair as alt 
import plotly.graph_objects as go 
import PIL  
import plotly.express as px 



st.title("Applying Models")

############################## Session states 

if "button_main" not in st.session_state:
    st.session_state["button_main"]=False

if "button_sec" not in st.session_state:
    st.session_state["button_sec"]=False

if "button_model" not in st.session_state:
    st.session_state["button_model"]=False

if "button_model_select" not in st.session_state:
    st.session_state["button_model_select"]=False

if "button_model_train" not in st.session_state:
    st.session_state["button_model_train"]=False

if "button_predict" not in st.session_state:
    st.session_state["button_predict"]=False


df=pd.read_csv("df_heart_clean_2.csv")


st.markdown("Based on the feature analysis from the previous page, select features to include in the model")
st.markdown("Leave as it is to select all features")

list_features=st.multiselect("Pick Features", 
                             ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"],
                             ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
                             )

st.write("The Following Features are being selected for the model:", list_features)



if st.button("Click when selection is finished and ready to continue"):
    st.session_state["button_main"]= not st.session_state["button_main"]


st.text("")
st.markdown("***")
st.text("")


if st.session_state["button_main"]:
    #Slecting continous and categorical variable 
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
    
    
    
    ########################################### Continous, Categorical, and output 
    cont=choose_cont(list_features, "cont")
    
    cat=choose_cont(list_features, "cat")
    
    output=["num"]

    st.markdown("Additional Preprocessing of Data")
    st.write("The first process is to apply Standard Scalar to regularize the data. Second process is to fix the imbalnce of the data.")

    X_cont=df[cont]
    y=df["num"]
    y=df[output].astype("category").to_numpy()


    col=cont+["num"]
    for_fig=pd.DataFrame((np.concatenate((X_cont, y), axis=1)), columns=col)
    x_str=col[0]
    
    y_str=col[1]
    

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
        

################################################################################################## To Preprocessing Stage 

if st.session_state["button_main"]:
     if st.button("Click to Preprocess the data"):
         st.session_state["button_sec"]= not st.session_state["button_sec"]

st.text("")
st.markdown("***")
st.text("")

if st.session_state["button_sec"]:
    st.write("Data is preprocessed. See the graphs below")
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





    

    st.markdown("Time to train model")



    
################## click button model 
if st.session_state["button_main"] and st.session_state["button_sec"]:
    if st.button("Train Model"):
        st.session_state["button_model"]=not st.session_state["button_model"]

st.text("")
st.markdown("***")
st.text("")

if st.session_state["button_model"]:
    
    st.write("6 models are available")

    model_option=st.selectbox(
        "Select model to train over the preprocessed data",
        ("ANN", "Random Forest", "Decision Tree", "KNN", "SVM", "Logistic Regression"))
    
#mode_cont=st.button("Click when done selecting model")


if st.session_state["button_main"] and st.session_state["button_sec"] and st.session_state["button_model"]:
    if st.button("Click to choose parameter for the choosen model"):
        st.session_state["button_model_select"]= not st.session_state["button_model_select"]

st.text("")
st.markdown("***")
st.text("")

if st.session_state["button_model_select"]:
    if model_option=="ANN":
        optimizer_input=st.text_input("Optimizer Choice", "adam")
        st.write("choise of optimizer", optimizer_input)
        no_epoch=int(st.number_input("Insert number of Epoch for training and press Enter"))
        st.write("Your choise of Epoch is ", no_epoch)
    elif model_option=="Random Forest":
        optimizer_input=st.text_input("Optimizer Choiiiiiiiiiiice", "adam")
        st.write("choise of optimizer", optimizer_input)
        no_epoch=int(st.number_input("Insert number of Epoch for training"))
        st.write("", no_epoch)



st.text("")
st.markdown("***")
st.text("")

######################################################################## Training model 
if st.session_state["button_main"] and st.session_state["button_sec"] and st.session_state["button_model"] and st.session_state["button_model_select"]:
    if st.button("Click to train model with given parameters"):
        st.session_state["button_model_train"]=not st.session_state["button_model_train"]


if st.session_state["button_model_train"]:
    st.markdown("Instruction:")
    st.markdown("**TO CHANGE MODEL** click the above button and close the training window. Then, choose another one from the selection box!")
    st.markdown(''':black[To change parameters and re-train model: **click the above button first**, then change parameters, then click it.]''')
    
    st.text("")
    st.markdown(''':red[TRAINING MODEL WAIT....!]''')
    
    if model_option=="ANN":
        ann_1=tf.keras.models.Sequential(
                [tf.keras.layers.Dense(units=6, activation="relu", input_dim=len(X_res[1,:])),
                tf.keras.layers.Dense(units=6, activation="relu"),
                tf.keras.layers.Dense(units=1, activation="sigmoid")])

        ann_1.compile(optimizer=optimizer_input,loss="binary_crossentropy",metrics=['accuracy'])

       
        ann_1.fit(X_res, y_res,batch_size=32,epochs=no_epoch)
          
        y_pred_test=ann_1.predict(X_test)
        y_pred_test=[0 if i<0.5 else 1 for i in y_pred_test]
        score_test=accuracy_score(y_pred_test, y_test)
        score_test=str(math.floor(score_test*100))+"%"
          
        y_pred_train=ann_1.predict(X_res)
        y_pred_train=[0 if i<0.5 else 1 for i in y_pred_train]
        score_train=accuracy_score(y_pred_train, y_res)
        score_train=str(math.floor(score_train*100))+"%"

        st.metric("Accuracy over Training data", score_train)
        st.metric("Accuracy over Test data", score_test)
        
    elif model_option=="Random Forest":
        st.write("RF to be developed")
        
    elif model_option=="Decision Tree":
        st.write("DT to be developed")

    elif model_option=="KNN":
        st.write("KNN yet to be developed")

    elif model_option=="SVM":
        st.write("SVM yet to be developed")
        
    else:
        st.write("Logistic Regression yet to be developed")
    
    

    


            
