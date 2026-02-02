import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Title and description
st.title("Wine Class Prediction App")
st.write("""
This app predicts the **Wine Class** based on its chemical composition 
using data derived from the source dataset.
""")

# Load data - The sources identify columns: WineClass, Alcohol, MalicAcid, Ash, 
# AlkalinityAsh, Mg, Phenols, Flavanoids, NonFlavanoidPhenols, Proanthocyanins, 
# Color, Hue, ODRatio, and Proline [1].
@st.cache_data
def load_data():
    # This assumes the dataset from the sources is saved as 'wine.csv'
    return pd.read_csv("wine.csv")

try:
    df = load_data()
    X = df.drop('WineClass', axis=1)
    y = df['WineClass']

    # Model training
    model = RandomForestClassifier()
    model.fit(X, y)

    # Sidebar inputs for features identified in source [1]
    st.sidebar.header("Input Wine Features")

    def user_input_features():
        # Note: The min/max ranges for these sliders are estimated from the 
        # samples provided in the sources [1-13] and may require independent verification.
        alcohol = st.sidebar.slider('Alcohol', 11.0, 15.0, 13.0)
        malic_acid = st.sidebar.slider('MalicAcid', 0.7, 5.8, 2.3)
        ash = st.sidebar.slider('Ash', 1.3, 3.3, 2.3)
        alkalinity_ash = st.sidebar.slider('AlkalinityAsh', 10.6, 30.0, 19.0)
        mg = st.sidebar.slider('Mg (Magnesium)', 70, 162, 100)
        phenols = st.sidebar.slider('Phenols', 0.9, 4.0, 2.3)
        flavanoids = st.sidebar.slider('Flavanoids', 0.3, 5.1, 2.0)
        non_flavanoid_phenols = st.sidebar.slider('NonFlavanoidPhenols', 0.1, 0.7, 0.3)
        proanthocyanins = st.sidebar.slider('Proanthocyanins', 0.4, 3.6, 1.5)
        color = st.sidebar.slider('Color Intensity', 1.2, 13.0, 5.0)
        hue = st.sidebar.slider('Hue', 0.4, 1.8, 1.0)
        od_ratio = st.sidebar.slider('ODRatio', 1.2, 4.0, 2.6)
        proline = st.sidebar.slider('Proline', 270, 1680, 750)
        
        data = {
            'Alcohol': alcohol, 'MalicAcid': malic_acid, 'Ash': ash,
            'AlkalinityAsh': alkalinity_ash, 'Mg': mg, 'Phenols': phenols,
            'Flavanoids': flavanoids, 'NonFlavanoidPhenols': non_flavanoid_phenols,
            'Proanthocyanins': proanthocyanins, 'Color': color, 'Hue': hue,
            'ODRatio': od_ratio, 'Proline': proline
        }
        return pd.DataFrame(data, index=)

    input_df = user_input_features()

    # Display user input
    st.subheader("User Input Parameters")
    st.write(input_df)

    # Prediction
    # The target classes found in the sources are: barolo, grignolino, and barbera [1-3].
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction")
    st.write(f"The predicted wine class is: **{prediction}**")

    st.subheader("Prediction Probability")
    st.write(pd.DataFrame(prediction_proba, columns=model.classes_))

except FileNotFoundError:
    st.error("Please ensure 'wine.csv' is in the same directory as this script.")