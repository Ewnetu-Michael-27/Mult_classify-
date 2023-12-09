#KNN
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
from sklearn import neighbors 


st.title("Training KNN Model and Predicting CAD")
image=Image.open("KNN.png")
st.image(image)

if "button_1" not in st.session_state:
    st.session_state["button_1"]=False

if "button_pre" not in st.session_state:
    st.session_state["button_pre"]=False

df=pd.read_csv("df_heart_clean_2.csv")


st.text("")

st.write("Choosing Parameters")
st.text("")
st.write("Input below number of neighbors to use.")
no_neigh=st.slider("Insert number of neighbors", 1,15,3)
st.write("Your choice of value is", no_neigh)
st.text("")
st.write("Choose percentage for Testing data in Train-Test data split")
portion=st.slider("Choose portion", 0.0,1.0,0.2)
st.write("Your choice of value is", portion)





#chose features and click button when done 

st.text("")
st.markdown("***")
st.text("")

st.markdown("Based on the feature analysis from the previous page, select features to include in the model")
st.markdown("You can leave as it is and click the button below to select all features")

list_features=st.multiselect("Pick Features", 
                             ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"],
                             ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
                             )


if st.button("Click to Train model"):
    st.session_state["button_1"]= not st.session_state["button_1"]

st.text("")
st.markdown("***")
st.text("")

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

    classifier_KNN=neighbors.KNeighborsClassifier(no_neigh)

    @st.cache_data
    def run_model():
        classifier_KNN.fit(X_res,y_res)
        return classifier_KNN
    
    model=run_model()

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

    keys=cont+cat
    values=[272727]*len(keys)

    key_val_pairs=zip(keys, values)
    dict_val=dict(key_val_pairs)

    st.write("Input value for age")
    age=st.slider("Input age", 0,130,25)
    st.write("The selected age value is ",age)
    dict_val["age"]=age

    st.text("")

    st.write("Select value for sex 1 for male and 0 for female")
    sex=float(st.selectbox("Select sex.", 
                    ("1", "0")))
    st.write("The selected age value is ",sex)
    dict_val["sex"]=sex

    st.text("")

    st.write("Input value for cp. 1 for Typical Angina, 2 Atypical Angina, 3 Non-Anginal Pain, and 4 asymptomatic")
    cp=float(st.selectbox("Select CP.", 
                    ("1", "2", "3", "4")))
    st.write("The selected cp value is ",cp)
    dict_val["cp"]=cp

    st.text("")

    st.write("Input value for Resting Blood Pressure in mm of Hg at admission to Hospital")
    trestbps=st.slider("Input Trestbps", 60.0,240.0,140.0)
    st.write("The selected trestbps value is ",trestbps)
    dict_val["trestbps"]=trestbps

    st.text("")
    st.write("Input value for Serum Cholestrol in mg/dl")
    chol=st.slider("Input Chol", 50.0,690.0,250.0)
    st.write("The selected chol value is ",chol)
    dict_val["chol"]=chol

    st.text("")

    st.write("Select value for Fasting Bloog sugar >120 1 is True and 0 is False")
    fbs=float(st.selectbox("Select fbs.", 
                    ("1", "0")))
    st.write("The selected fbs value is ",fbs)
    dict_val["fbs"]=fbs
    st.text("")

    st.markdown("Select value for Resting Electrocardiographic Results. 0 Normal, 1 Having ST-T wave abnormality, 2 showing probable or definite left ventricular hypertrophy")
    restecg=float(st.selectbox("Select restecg.", 
                    ("0", "1", "2")))
    st.write("The selected restecg value is ",restecg)
    dict_val["restecg"]=restecg 

    st.text("")
    st.write("Input value for thalach. Maximum Heart Rate achieved")
    thalach=st.slider("Input Thala,ch", 40.0,220.0,140.0)
    st.write("The selected thalach value is ",thalach)
    dict_val["thalach"]=thalach 

    st.text("")

    st.write("Input value for excercise induced angina. 1 for yes and 0 for no")
    exang=float(st.selectbox("Select exang.", 
                    ("0", "1")))
    st.write("The selected exang value is ",exang)
    dict_val["exang"]=exang 

    st.text("")
    st.write("Input value for oldpeak. ST Depression induced by excercise relative to rest")
    oldpeak=st.slider("Input Thala,ch", 0.0,9.0,3.0)
    st.write("The selected oldpeak value is ",oldpeak)
    dict_val["oldpeak"]=oldpeak

    st.text("")
    st.write("Input value for for the slope of peak exercise ST segment. 1 upsloping, 2 Flat, 3 Down sloping")
    slope=float(st.selectbox("Select slope.", 
                    ("1", "2", "3")))
    st.write("The selected slope value is ",slope)
    dict_val["slope"]=slope

    st.text("")
    st.write("Input value for number of major vessels colored by fluroscopy")
    ca=float(st.selectbox("Select ca.", 
                    ("0", "1", "2", "3")))
    st.write("The selected ca value is ",ca)
    dict_val["ca"]=ca 

    st.text("")
    st.write("Input value for Thallium scintigraphy, 3 normal 6 Fixed defect 7 Reversable defect")
    thal=float(st.selectbox("Select thal.", 
                    ("3", "5", "6", "7")))
    st.write("The selected thal value is ",thal)
    dict_val["thal"]=thal           


    st.text("")
    st.write("Summary of selected values")
    st.write(dict_val)

    p_values=[(list(dict_val))]

    def predict_output(model, list_v, classifier):
        list_v=np.array(list_v)
        if classifier=="ANN":
            p_a=model.predict(list_v)[0][0]
            return "CAD is Present" if p_a>0.5 else "CAD Is Absent"
        else:
            return model.predict(list_v)[0]
        
    button_pre=st.button("Click to Predict")

    if button_pre:
        st.write("Checking")