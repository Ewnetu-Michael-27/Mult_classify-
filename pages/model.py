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
from PIL import Image
import plotly.express as px 
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors 
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 



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
st.markdown("You can leave as it is and click the button below to select all features")

list_features=st.multiselect("Pick Features", 
                             ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"],
                             ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
                             )

st.write("The Following Features are being selected for the model:", list_features)



if st.button("Click when selection is finished or to roll back page below"):
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
     if st.button("Click to Preprocess the data or to roll back page below"):
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
    if st.button("Click to train model or to roll back page below"):
        st.session_state["button_model"]=not st.session_state["button_model"]

st.text("")
st.markdown("***")
st.text("")

if st.session_state["button_model"]:
    
    st.write("5 models are available")

    model_option=st.selectbox(
        "Select model to train over the preprocessed data",
        ("ANN", "Random Forest", "KNN", "SVM", "Logistic Regression"))
    
#mode_cont=st.button("Click when done selecting model")


if st.session_state["button_main"] and st.session_state["button_sec"] and st.session_state["button_model"]:
    if st.button("Click to choose parameter for the choosen model or to roll back page below"):
        st.session_state["button_model_select"]= not st.session_state["button_model_select"]

st.text("")
st.markdown("***")
st.text("")

if st.session_state["button_model_select"]:
    if model_option=="ANN":
        st.write("You have choosen Artificial Neural Network from TensorFlow")
        image=Image.open("ANN.png")
        st.image(image)
        st.write("Write Optimizer below. Options are (adam, rmsprop, ftrl, adadelta, adafactor, sgd, nadam, lion) **adam** is recommended")
        optimizer_input=st.text_input("Optimizer Choice ", "adam")
        st.write("choice of optimizer", optimizer_input)
        st.text("")
        st.write("Insert number of Epochs below. **500** is recommended")
        no_epoch=int(st.number_input("Insert number of Epoch for training and press Enter"))
        st.write("Your choice of Epoch is ", no_epoch)
    elif model_option=="Random Forest":
        st.write("You have choosen Random Forest Model from Sklearn")
        #st.image("RF.png")
        image=Image.open("RF.png")
        st.image(image)
        st.write("Input below n_estimators, which is the number of trees in the forest. **Below 10** is recommended")
        no_n_est=int(st.number_input("Insert n_estimators and press Enter."))
        st.write("The choice of n_estimators is ", no_n_est)
        st.text("")
        st.write("Write below choice of criterion to measure quality of split. Options are (gini, entropy, log_loss ). **gini** is recommended!")
        crit=st.text_input("Choice of criterion", "gini")
        st.write("Choice of criterion is ", crit)
    elif model_option=="KNN":
        st.write("You have selected KNN model for classification.")
        #st.image("KNN.png")
        image=Image.open("KNN.png")
        st.image(image)
        st.write("Input below number of neighbors to use.")
        no_neigh=int(st.number_input("Insert number of neighbors"))
        st.write("Your choice of value is", no_neigh)
    elif model_option=="SVM":
        st.write("You have choosen Support Vector Machine: SVM from sklearn")
        #st.image("SVM.PNG")
        image=Image.open("SVM.png")
        st.image(image)
        st.write("Write below which kernel to use. Options are linear, poly, sigmoid, precomputed")
        kernel=st.text_input("Write the kernel below and press enter", "linear")
        st.write("You have selected", kernel)
        st.text("")
        st.write("Choose Regularization parameter below. 1 is recommended")
        c_reg=st.number_input("Insert C and press enter")
        st.write("Your choice of C is ", c_reg)
    elif model_option=="Logistic Regression":
        st.write("You have choosen Logistic Regression from sklearn")
        #st.image("LR.PNG")
        image=Image.open("LR.png")
        st.image(image)
        st.write("write choice of solver below. Options are liblinear, lbfgs, sag, saga, newton-cholesky. liblinear and lbfgs are highly recommended")
        solver=st.text_input("Write choice of solver and press enter", "liblinear")
        st.write("Your choice of solver is", solver)
        st.text("")
        st.write("Choose maximum number of iteration below. 100 is recommended")
        max_iter=int(st.number_input("Input choice of maximum number of iteration and press enter"))
        st.write("Your choice of value is", max_iter)



st.text("")
st.markdown("***")
st.text("")

######################################################################## Training model 
if st.session_state["button_main"] and st.session_state["button_sec"] and st.session_state["button_model"] and st.session_state["button_model_select"]:
    if st.button("Click to train model with given parameters or to roll back page below"):
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


        confusion_matrix_1=confusion_matrix(y_test, y_pred_test)

        z = np.flip(confusion_matrix_1,0)
        x = ['Predict 1', 'Prdict 0']
        y =  ['True 0', 'True 1']

        # change each element of z to type string for annotations
        z_text = [[str(x) for x in y] for y in z]

        # set up figure 
        fig = ff.create_annotated_heatmap(z, y=y, x=x, annotation_text=z_text, colorscale='Viridis')

        # add title
        fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                  #xaxis = dict(title='x'),
                  #yaxis = dict(title='x')
                 )

        # add custom xaxis title
        fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=0.5,
                        y=-0.15,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))

        # add custom yaxis title
        fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.35,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))

        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=50, l=200))

        # add colorbar
        fig['data'][0]['showscale'] = True
        st.plotly_chart(fig)
        
    elif model_option=="Random Forest":
        Classifier_RF=RandomForestClassifier(n_estimators=no_n_est, criterion=crit)
        Classifier_RF.fit(X_res,y_res)

        y_pred_test=Classifier_RF.predict(X_test)
        score_test=accuracy_score(y_pred_test, y_test)
        score_test=str(math.floor(score_test*100))+"%"
          
        y_pred_train=Classifier_RF.predict(X_res)
        score_train=accuracy_score(y_pred_train, y_res)
        score_train=str(math.floor(score_train*100))+"%"

        st.metric("Accuracy over Training data", score_train)
        st.metric("Accuracy over Test data", score_test)

        confusion_matrix_2=confusion_matrix(y_test, y_pred_test)

        z = np.flip(confusion_matrix_2,0)
        x = ['Predict 1', 'Prdict 0']
        y =  ['True 0', 'True 1']

        # change each element of z to type string for annotations
        z_text = [[str(x) for x in y] for y in z]

        # set up figure 
        fig_2 = ff.create_annotated_heatmap(z, y=y, x=x, annotation_text=z_text, colorscale='Viridis')

        # add title
        fig_2.update_layout(title_text='<i><b>Confusion matrix on The Test Data</b></i>',
                  #xaxis = dict(title='x'),
                  #yaxis = dict(title='x')
                 )

        # add custom xaxis title
        fig_2.add_annotation(dict(font=dict(color="black",size=14),
                        x=0.5,
                        y=-0.15,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))

        # add custom yaxis title
        fig_2.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.35,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))

        # adjust margins to make room for yaxis title
        fig_2.update_layout(margin=dict(t=50, l=200))

        # add colorbar
        fig_2['data'][0]['showscale'] = True
        st.plotly_chart(fig_2)

    elif model_option=="KNN":
        
        classifier_KNN=neighbors.KNeighborsClassifier(no_neigh)
        classifier_KNN.fit(X_res, y_res)

        y_pred_test=classifier_KNN.predict(X_test)
        score_test=accuracy_score(y_pred_test, y_test)
        score_test=str(math.floor(score_test*100))+"%"
          
        y_pred_train=classifier_KNN.predict(X_res)
        score_train=accuracy_score(y_pred_train, y_res)
        score_train=str(math.floor(score_train*100))+"%"

        st.metric("Accuracy over Training data", score_train)
        st.metric("Accuracy over Test data", score_test)

        confusion_matrix_3=confusion_matrix(y_test, y_pred_test)

        z = np.flip(confusion_matrix_3,0)
        x = ['Predict 1', 'Prdict 0']
        y =  ['True 0', 'True 1']

        # change each element of z to type string for annotations
        z_text = [[str(x) for x in y] for y in z]

        # set up figure 
        fig_3 = ff.create_annotated_heatmap(z, y=y, x=x, annotation_text=z_text, colorscale='Viridis')

        # add title
        fig_3.update_layout(title_text='<i><b>Confusion matrix on The Test Data</b></i>',
                  #xaxis = dict(title='x'),
                  #yaxis = dict(title='x')
                 )

        # add custom xaxis title
        fig_3.add_annotation(dict(font=dict(color="black",size=14),
                        x=0.5,
                        y=-0.15,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))

        # add custom yaxis title
        fig_3.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.35,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))

        # adjust margins to make room for yaxis title
        fig_3.update_layout(margin=dict(t=50, l=200))

        # add colorbar
        fig_3['data'][0]['showscale'] = True
        st.plotly_chart(fig_3)



    elif model_option=="SVM":

        classifier_SVM=svm.SVC(kernel=kernel, C=c_reg)
        classifier_SVM.fit(X_res, y_res)

        y_pred_test=classifier_SVM.predict(X_test)
        score_test=accuracy_score(y_pred_test, y_test)
        score_test=str(math.floor(score_test*100))+"%"
          
        y_pred_train=classifier_SVM.predict(X_res)
        score_train=accuracy_score(y_pred_train, y_res)
        score_train=str(math.floor(score_train*100))+"%"

        st.metric("Accuracy over Training data", score_train)
        st.metric("Accuracy over Test data", score_test)

        confusion_matrix_4=confusion_matrix(y_test, y_pred_test)

        z = np.flip(confusion_matrix_4,0)
        x = ['Predict 1', 'Prdict 0']
        y =  ['True 0', 'True 1']

        # change each element of z to type string for annotations
        z_text = [[str(x) for x in y] for y in z]

        # set up figure 
        fig_4 = ff.create_annotated_heatmap(z, y=y, x=x, annotation_text=z_text, colorscale='Viridis')

        # add title
        fig_4.update_layout(title_text='<i><b>Confusion matrix on The Test Data</b></i>',
                  #xaxis = dict(title='x'),
                  #yaxis = dict(title='x')
                 )

        # add custom xaxis title
        fig_4.add_annotation(dict(font=dict(color="black",size=14),
                        x=0.5,
                        y=-0.15,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))

        # add custom yaxis title
        fig_4.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.35,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))

        # adjust margins to make room for yaxis title
        fig_4.update_layout(margin=dict(t=50, l=200))

        # add colorbar
        fig_4['data'][0]['showscale'] = True
        st.plotly_chart(fig_4)


        
    else:
        classifier_LR=LogisticRegression(solver=solver, max_iter=max_iter)
        classifier_LR.fit(X_res, y_res)
        

        y_pred_test=classifier_LR.predict(X_test)
        score_test=accuracy_score(y_pred_test, y_test)
        score_test=str(math.floor(score_test*100))+"%"
          
        y_pred_train=classifier_LR.predict(X_res)
        score_train=accuracy_score(y_pred_train, y_res)
        score_train=str(math.floor(score_train*100))+"%"

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
    
    



st.write("Based on the trained model, input values below to predict presence or absence of CAD")

if st.session_state["button_main"] and st.session_state["button_sec"] and st.session_state["button_model"] and st.session_state["button_model_select"] and st.session_state["button_model_train"]:
    if st.button("Click to predict from values or to roll back page below"):
        st.session_state["button_predict"]=not st.session_state["button_predict"]

if st.session_state["button_predict"]:

    st.write("Input values below. **Make sure to only insert values for the features selected above**. If the particular feature is not selected, just leave it")

    keys=cont+cat
    values=[3129]*len(keys)

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



    st.write(dict_val)

    #def transformer(dict_v):


    #predicting values now

    p_values=[(list(dict_val))]

    def predict_output(model, list_v, classifier):
        list_v=np.array(list_v)
        if classifier=="ANN":
            p_a=model.predict(list_v)[0][0]
            return "CAD is Present" if p_a>0.5 else "CAD Is Absent"
        else:
            return model.predict(list_v)[0]
        
    st.write("The Prediction value is")
    

            



    

            
