#LR
import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import math 
import altair as alt 
import plotly.graph_objects as go 
from PIL import Image
import plotly.express as px 
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
from sklearn.linear_model import LogisticRegression 

if "button_1" not in st.session_state:
    st.session_state["button_1"]=False


if "button_pre" not in st.session_state:
    st.session_state["button_pre"]=False


st.title("Training a Logistic Regression Model and Predicting CAD")

df=pd.read_csv("df_heart_clean_2.csv")

st.text("")

image=Image.open("LR.png")
st.image(image)
st.text("")

st.write("Choose a solver below. . liblinear and lbfgs are highly recommended")
solver=st.selectbox("Choose a solver",
                    ("liblinear", "lbfgs", "sag", "saga", "newton-cholesky"))
st.write("Your choice of solver is", solver)
st.text("")
st.write("Choose maximum number of iteration below. 100 is recommended")
max_iter=st.slider("Choose maximum number of iteration", 1,1000,100)
st.write("Your choice of value is", max_iter)
st.write("Choose percentage for Testing data in Train-Test data split")
portion=st.slider("Choose portion", 0.0,1.0,0.2)
st.write("Your choice of value is", portion)

st.text("")
clear_cac=st.button("Click if values above are altered")

if clear_cac:
    st.cache_data.clear()


st.text("")
st.markdown("***")
st.text("")

st.markdown("Based on the feature analysis from the previous page, select features to include in the model")
st.markdown("You can leave as it is and click the button below to select all features")

list_features=st.multiselect("Pick Features", 
                             ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"],
                             ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
                             )


st.text("")
st.markdown("***")
st.text("")

if st.button("Click to Train model"):
    st.session_state["button_1"]= not st.session_state["button_1"]


if st.session_state["button_1"]:
    st.markdown(''':red[TRAINING MODEL WAIT....!]''')
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
    output=["num"]

    X_cont=df[cont].to_numpy()
    y=df[output].astype("category").to_numpy()

    
    my_scaler = StandardScaler()
    my_scaler.fit(X_cont)
    X_cont_sc=my_scaler.transform(X_cont)

    X_cat=df[cat].to_numpy()
    X_new=np.hstack((X_cont_sc, X_cat))

    X_train,X_test,y_train,y_test = train_test_split(X_new,y,test_size=portion,random_state=42)

    samp = {0: len(y_train[y_train == 0]), 1: len(y_train[y_train == 0])}
    sampler=RandomOverSampler(sampling_strategy=samp, random_state=42)
    X_res, y_res=sampler.fit_resample(X_train, y_train)


    classifier_LR=LogisticRegression(solver=solver, max_iter=max_iter)


    @st.cache_data
    def run_model(_m, data, out):
        _m.fit(data, out)
        return _m
    
    model=run_model(classifier_LR,X_res, y_res)
    

    y_pred_test=model.predict(X_test)
    score_test=accuracy_score(y_pred_test, y_test)
    score_test=str(math.floor(score_test*100))+"%"
        
    y_pred_train=model.predict(X_res)
    score_train=accuracy_score(y_pred_train, y_res)
    score_train=str(math.floor(score_train*100))+"%"

    st.write("**MODEL TRAINED**")
    st.write("**See the two tabs below: Result, and Prediction**")

    st.text("")
    st.markdown("***")
    st.text("")

    
    st.metric("Accuracy over Training data", score_train)
    st.metric("Accuracy over Test data", score_test)

    confusion_matrix_5=confusion_matrix(y_test, y_pred_test)

    z = np.flip(confusion_matrix_5,0)
    x = ['Predict 1', 'Prdict 0']
    y =  ['True 0', 'True 1']

    # change each element of z to type string for annotations
    z_text = [[str(x) for x in y] for y in z]

    # set up figure 
    fig_5 = ff.create_annotated_heatmap(z, y=y, x=x, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig_5.update_layout(title_text='<i><b>Confusion matrix on The Test Data</b></i>',
                #xaxis = dict(title='x'),
                #yaxis = dict(title='x')
                )

    # add custom xaxis title
    fig_5.add_annotation(dict(font=dict(color="black",size=14),
                    x=0.5,
                    y=-0.15,
                    showarrow=False,
                    text="Predicted value",
                    xref="paper",
                    yref="paper"))

    # add custom yaxis title
    fig_5.add_annotation(dict(font=dict(color="black",size=14),
                    x=-0.35,
                    y=0.5,
                    showarrow=False,
                    text="Real value",
                    textangle=-90,
                    xref="paper",
                    yref="paper"))

    # adjust margins to make room for yaxis title
    fig_5.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig_5['data'][0]['showscale'] = True
    st.plotly_chart(fig_5)


if st.session_state["button_1"]:
    if st.button("Click to Predict cad from input values"):
        st.session_state["button_pre"]=not st.session_state["button_pre"]

if st.session_state["button_pre"]:
    st.write("Input values below. **Make sure to only insert values for the features selected above**. If the particular feature is not selected, just leave it")

    
    dict_val={}

    st.text("")
    st.text("")

    if "age" in list_features:
        st.write("Input value for age")
        age=st.slider("Input age", 0,130,66)
        st.write("The selected age value is ",age)
        dict_val["age"]=age

    st.text("")

    if "sex" in list_features:
        st.write("Select value for sex 1 for male and 0 for female")
        sex=float(st.selectbox("Select sex.", 
                        ("1", "0")))
        st.write("The selected age value is ",sex)
        dict_val["sex"]=sex

    st.text("")
    
    if "cp" in list_features:
        st.write("Input value for cp. 1 for Typical Angina, 2 Atypical Angina, 3 Non-Anginal Pain, and 4 asymptomatic")
        cp=float(st.selectbox("Select CP.", 
                        ("4", "3", "2", "1")))
        st.write("The selected cp value is ",cp)
        dict_val["cp"]=cp

    st.text("")

    if "trestbps" in list_features:
        st.write("Input value for Resting Blood Pressure in mm of Hg at admission to Hospital")
        trestbps=st.slider("Input Trestbps", 60.0,240.0,160.0)
        st.write("The selected trestbps value is ",trestbps)
        dict_val["trestbps"]=trestbps

    st.text("")

    if "chol" in list_features:
        st.write("Input value for Serum Cholestrol in mg/dl")
        chol=st.slider("Input Chol", 50.0,690.0,286.0)
        st.write("The selected chol value is ",chol)
        dict_val["chol"]=chol

    st.text("")

    if "fbs" in list_features:
        st.write("Select value for Fasting Bloog sugar >120 1 is True and 0 is False")
        fbs=float(st.selectbox("Select fbs.", 
                        ("0", "1")))
        st.write("The selected fbs value is ",fbs)
        dict_val["fbs"]=fbs
        st.text("")

    if "restecg" in list_features:
        st.markdown("Select value for Resting Electrocardiographic Results. 0 Normal, 1 Having ST-T wave abnormality, 2 showing probable or definite left ventricular hypertrophy")
        restecg=float(st.selectbox("Select restecg.", 
                        ("2", "1", "0")))
        st.write("The selected restecg value is ",restecg)
        dict_val["restecg"]=restecg 

    st.text("")

    if "thalach" in list_features:
        st.write("Input value for thalach. Maximum Heart Rate achieved")
        thalach=st.slider("Input Thala,ch", 40.0,220.0,108.0)
        st.write("The selected thalach value is ",thalach)
        dict_val["thalach"]=thalach 

    st.text("")

    if "exang" in list_features:
        st.write("Input value for excercise induced angina. 1 for yes and 0 for no")
        exang=float(st.selectbox("Select exang.", 
                        ("1", "0")))
        st.write("The selected exang value is ",exang)
        dict_val["exang"]=exang 

    st.text("")

    if "oldpeak" in list_features:
        st.write("Input value for oldpeak. ST Depression induced by excercise relative to rest")
        oldpeak=st.slider("Input Thala,ch", 0.0,9.0,1.5)
        st.write("The selected oldpeak value is ",oldpeak)
        dict_val["oldpeak"]=oldpeak

    st.text("")

    if "slope" in list_features:
        st.write("Input value for for the slope of peak exercise ST segment. 1 upsloping, 2 Flat, 3 Down sloping")
        slope=float(st.selectbox("Select slope.", 
                        ("3.0", "2.0", "1.0")))
        st.write("The selected slope value is ",slope)
        dict_val["slope"]=slope

    st.text("")

    if "ca" in list_features:
        st.write("Input value for number of major vessels colored by fluroscopy")
        ca=float(st.selectbox("Select ca.", 
                        ("3", "2", "1", "0")))
        st.write("The selected ca value is ",ca)
        dict_val["ca"]=ca 

    st.text("")

    if "thal" in list_features:
        st.write("Input value for Thallium scintigraphy, 3 normal 6 Fixed defect 7 Reversable defect")
        thal=float(st.selectbox("Select thal.", 
                        ("3", "5", "6", "7")))
        st.write("The selected thal value is ",thal)
        dict_val["thal"]=thal           


    st.text("")
    st.write("Summary of selected values")
    st.write(dict_val)
        
        
    button_pre=st.button("Click to Predict")
    if button_pre:
        #transform the continous values from the inputs 
        cont_predict=[]

        for i in list(dict_val):
            if i in cont:
                cont_predict.append(dict_val[i])

        #time to transfrom 
        cont_predict_sc=my_scaler.transform([cont_predict])

        #adding the categorical valriables

        for i in list(dict_val):
            if i in cat:
                val=dict_val[i]
                cont_predict_sc=np.append(cont_predict_sc, [val])
        
        #time for prediction
        p_a=model.predict(cont_predict_sc.reshape((1,13)))[0]
        st.metric("Probability of CAD Presence", p_a)
        if p_a>0.5:
            st.write("**CAD is present**")
            st.write("Please consult your primary care physician. Click [here](https://www.cdc.gov/heartdisease/coronary_ad.htm) to read more!")
        else:
            st.write("**CAD Is Absent.**")
            st.write("You are safe. However, please read [here](https://www.cdc.gov/heartdisease/coronary_ad.htm) more to increase awarness.")
            






