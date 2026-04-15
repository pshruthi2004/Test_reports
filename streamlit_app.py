import streamlit as st
import pickle

st.title("Multi Hospital Lab Result Prediction")

with open("model.pkl", "rb") as f:
    model, le_hospital, le_test, le_unit = pickle.load(f)

hospital = st.selectbox("Select Hospital", le_hospital.classes_)
test = st.selectbox("Select Test", le_test.classes_)
unit = st.selectbox("Select Unit", le_unit.classes_)

if st.button("Predict"):
    h = le_hospital.transform([hospital])[0]
    t = le_test.transform([test])[0]
    u = le_unit.transform([unit])[0]

    pred = model.predict([[h, t, u]])
    st.success(f"Predicted Value: {pred[0]:.2f}")
