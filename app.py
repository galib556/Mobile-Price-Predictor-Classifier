import streamlit as st
import pickle
import numpy as np
import pandas as pd
@st.cache_resource
def load():
    return pickle.load(open("lr_model.pkl","rb"))


model=load()
labels={0:"Low Cost",1:"Medium Cost",2:"High Cost",3:"Very High Cost"}


st.title("Mobile Price Prediction Classifier")
battery=st.slider("Battery(mAh)",500,2000,1200)
px_h=st.slider("Pixel Height",0,2000,700)
px_w=st.slider("Pixel Width",500,2000,1200)
ram=st.slider("RAM(MB)",256,4000,2000)

if st.button("Predict"):
    X=pd.DataFrame([[battery,px_h,px_w,ram]],
         columns=['battery_power','px_height','px_width','ram'])
    probs=model.predict_proba(X)[0]
    for c,p in zip(model.classes_,probs):
        st.write(f"{labels[int(c)]}:{p:0}")
        st.progress(float(p))
    
    
    
