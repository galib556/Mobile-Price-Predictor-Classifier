import gradio as gr
import pickle
import numpy as np
import pandas as pd
with open("lr_model.pkl","rb") as file:
    model=pickle.load(file)

labels={0:"Low Cost",1:"Medium Cost",2:"High Cost",3:"Very High Cost"}


def predict(battery_power,px_height,px_width,ram):
    input_df=pd.DataFrame([
    [
        battery_power,px_height,px_width,ram
    ]
],columns=[
        'battery_power','px_height','px_width','ram'
    ])    
    prediction=model.predict_proba(input_df)[0]
    return {labels[int(c)]:float(p) for c,p in zip(model.classes_,prediction)}

inputs=[
    gr.Slider(500,2000,label="Battery(mAh)"),
    gr.Slider(0,2000,label="Pixel Height"),
    gr.Slider(500,2000,label="Pixel Width"),
    gr.Slider(256,4000,label="Ram(MB)")
]

app=gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Label(num_top_classes=4,label="Predicted Price Range"),
    title="Mobile Price Prediction Classifier"
)
app.launch()  
    
    
    
