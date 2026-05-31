import streamlit as st
import pandas as pd 
import joblib
#Loading Variation engine 
model= joblib.load('real_estate_brain.pkl')

#Configuring App Interface 
st.set_page_config(page_title="Real Estate Valuation", layout="centered")
st.title("Real Estate Asset Valuation Engine")
st.write("Input the structural parameters below to instantly caluculate the estimated market value of property.")

st.markdown("---")

#Create the input features(V2 Model)
st.subheader("Property Specifications")
col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("Square Footage (GrLivArea)", min_value=500, max_value=15000, value=2500, step=100)
    beds = st.number_input("Bedrooms Above Grade", min_value=1, max_value=10, value=4, step=1)
    year = st.number_input("Year Built", min_value=1800, max_value=2026, value=2022, step=1)

with col2:
    qualtiy = st.slider("Overall Finish Quality(1-10)", min_value=1, max_value=10, value=8)
    garage = st.slider("Garage Capacity (Cars)", min_value=0, max_value=6, value=2)

st.markdown("---")  
#The Execution Engine
if st.button("Calculate Market Value", type="primary"):
    input_features = [[sqft, beds, year, qualtiy, garage]]

    raw_prediction =  model.predict(input_features)[0]

    formatted_price = f"${raw_prediction:,.2f}"

    st.success(f"Estimated Asset Value: **{formatted_price}**")