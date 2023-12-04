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



st.title("Applying Models")


df=pd.read_csv("df_heart_clean_2.csv")


st.markdown("Based on the feature analysis from the previous page, select features to include in the model")
st.markdown("Leave as it is to select all features")

list_features=st.multiselect("Pick Features", 
                             ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"],
                             ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
                             )

#button_main=st.button("Click when done selecting")

if "button_main" not in st.session_state:
    st.session_state["button_main"]=False

if "button_sec" not in st.session_state:
    st.session_state["button_sec"]=False

if "button_model" not in st.session_state:
    st.session_state["button_model"]=False

if st.button("Click when done selecting"):
    st.session_state["button_main"]= not st.session_state["button_main"]

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
    
    cont=choose_cont(list_features, "cont")
    cat=choose_cont(list_features, "cat")

    st.markdown("Additional Preprocessing of Data")


if st.session_state["button_main"]:
     if st.button("Continue Preprocessing"):
         st.session_state["button_sec"]= not st.session_state["button_sec"]

if st.session_state["button_sec"]:
    X_cont=df[cont].to_numpy()
    my_scaler = StandardScaler()
    my_scaler.fit(X_cont)
    X_cont_sc=my_scaler.transform(X_cont)

    X_cat=df[cat].to_numpy()
    X_new=np.hstack((X_cont_sc, X_cat))
    output=["num"]
    y=df[output].astype("category").to_numpy()


    X_train,X_test,y_train,y_test = train_test_split(X_new,y,test_size=0.2,random_state=42)


    samp = {0: len(y_train[y_train == 0]), 1: len(y_train[y_train == 0])}
    sampler=RandomOverSampler(sampling_strategy=samp, random_state=42)
    X_res, y_res=sampler.fit_resample(X_train, y_train)

    st.markdown("Time to train model")
    

if st.session_state["button_main"] and st.session_state["button_sec"]:
    if st.button("Train Model"):
        st.session_state["button_model"]=not st.session_state["button_model"]



if st.session_state["button_model"]:
    
    st.text("6 models are available")

    model_option=st.selectbox(
        "Select model to train over the preprocessed data",
        ("ANN", "Random Forest", "Decision Tree", "KNN", "SVM", "Logistic Regression"))
    
    mode_cont=st.button("Click when done selecting model")

    if mode_cont:
        if model_option=="ANN":
            ann_1=tf.keras.models.Sequential(
                [tf.keras.layers.Dense(units=6, activation="relu", input_dim=len(X_res[1,:])),
                tf.keras.layers.Dense(units=6, activation="relu"),
                tf.keras.layers.Dense(units=1, activation="sigmoid")])

            ann_1.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
        
            ann_1.fit(X_res, y_res,batch_size=32,epochs=500)
          
            y_pred_test=ann_1.predict(X_test)
            y_pred_test=[0 if i<0.5 else 1 for i in y_pred_test]
            score_test=accuracy_score(y_pred_test, y_test)
            score_test=str(math.floor(score_test*100))+"%"
          
            y_pred_train=ann_1.predict(X_res)
            y_pred_train=[0 if i<0.5 else 1 for i in y_pred_train]
            score_train=accuracy_score(y_pred_train, y_res)
            score_train=str(math.floor(score_train*100))+"%"

            st.metric("Accuracy over Training data", score_train)
            st.text("And")
            st.metric("Accuracy over Test data", score_test)
        
        else:
            st.text("Model yet to be developed")
    
    

    


            
