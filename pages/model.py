import streamlit as st 
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




st.title("Applying Models")



df=pd.read_csv("df_heart_clean_2.csv")

pre=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
output=["num"]


X=df[pre].to_numpy()
y=df[output].astype("category").to_numpy()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


button=st.button("Train ANN")

if button:
    ann_1=tf.keras.models.Sequential(
    [tf.keras.layers.Dense(units=6, activation="relu", input_dim=len(X_train[1,:])),
     tf.keras.layers.Dense(units=6, activation="relu"),
     tf.keras.layers.Dense(units=1, activation="sigmoid")
        
    ]
    )
    ann_1.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
    ann_1.summary()
    ann_1.fit(X_train, y_train,batch_size=32,epochs=500)
    y_pred=ann_1.predict(X_test)
    y_pred=[0 if i<0.5 else 1 for i in y_pred]
    score=accuracy_score(y_pred, y_test)
    
    st.metric("Accuracy over Test data", score)
