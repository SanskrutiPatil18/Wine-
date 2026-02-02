import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(page_title="Wine Class Predictor", layout="wide")

st.title("Wine Class Prediction App")
st.write("""
This app predicts the **Wine Class** based on its chemical composition 
using attributes identified in the source dataset.
""")

# Function to load data and train the model
@st.cache_data
def load_and_train_model():
    # The source data contains the header: WineClass, Alcohol, MalicAcid, Ash, etc.
    # Ensure 'wine.csv' is in your directory
    df = pd.read_csv("wine.csv")
    
    # Cleaning data if there are trailing commas as seen in source samples [1, 4, 5]
    X = df.drop('WineClass', axis=1)
    y = df['WineClass']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns

# Sidebar for user inputs
def get_user_inputs():
    st.sidebar.header("Input Wine Features")
    
    # Ranges are derived from source data points (e.g., Mg up to 162, Proline up to 1680) [4, 6]
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
    return pd.DataFrame(data, index=) # FIXED: Index provided to prevent Syntax/Value errors

# Main Logic
try:
    # 1. Train Model
    model, feature_names = load_and_train_model()
    
    # 2. Get Input
    input_df = get_user_inputs()
    
    # 3. Display Parameters
    st.subheader("User Input Parameters")
    st.write(input_df)
    
    # 4. Predict
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # 5. Display Results
    st.subheader("Prediction Result")
    # Result will be one of the source classes: barolo, grignolino, or barbera
    st.success(f"The predicted wine class is: **{prediction.upper()}**")
    
    st.subheader("Prediction Probability")
    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
    st.write(proba_df)

except FileNotFoundError:
    st.error("Missing Dataset: Please upload 'wine.csv' to the same directory.")
except Exception as e:
    st.error(f"An error occurred: {e}")


