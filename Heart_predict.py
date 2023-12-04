import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.set_page_config(
    page_title="Prediction of CAD"
)

st.title("Applying Several Classification Models to Predict the Presence of CAD")

st.sidebar.success("Select a Page Above")


b_1=st.button("Check_1")

if b_1:
    st.markdown("Continue")
    b_2=st.button("Check_2")
    if b_2:
        st.markdown("Continue")
        b_3=st.button("Check_3")
        if b_3:
            st.markdown("Done")
            
            
        
    







