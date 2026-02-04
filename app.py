import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(page_title="Wine Class Predictor", layout="wide")

st.title("Wine Class Prediction App")
st.write("""
This app predicts the **Wine Class** based on chemical composition and allows you 
to explore the source dataset dynamically.
""")

# 1. Function to load data and train the model
@st.cache_data
def load_data_and_model():
    # Load dataset with headers identified in the sources [1]
    df = pd.read_csv("wine.csv")
    df = df.dropna() # Cleans up any trailing commas or missing values from source excerpts [1, 4]
    
    X = df.drop('WineClass', axis=1)
    y = df['WineClass']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return df, model

try:
    df, model = load_data_and_model()

    # --- SIDEBAR: DATASET DISPLAY OPTIONS ---
    st.sidebar.header("Dataset Display Settings")
    # User can choose how many rows to see (e.g., 5 or 8)
    num_rows = st.sidebar.number_input("Select number of rows to preview:", min_value=1, max_value=len(df), value=5)

    # --- SIDEBAR: PREDICTION INPUTS ---
    st.sidebar.header("Input Wine Features for Prediction")
    
    def get_user_inputs():
        # Feature ranges are based on source data (e.g., Mg up to 162, Proline up to 1680) [5, 6]
        alcohol = st.sidebar.slider('Alcohol', 11.0, 15.0, 13.0)
        malic_acid = st.sidebar.slider('MalicAcid', 0.7, 5.8, 2.3)
        ash = st.sidebar.slider('Ash', 1.3, 3.3, 2.3)
        alkalinity_ash = st.sidebar.slider('AlkalinityAsh', 10.6, 30.0, 19.0)
        mg = st.sidebar.slider('Mg', 70, 162, 100)
        phenols = st.sidebar.slider('Phenols', 0.9, 4.0, 2.3)
        flavanoids = st.sidebar.slider('Flavanoids', 0.3, 5.1, 2.0)
        non_flavanoid_phenols = st.sidebar.slider('NonFlavanoidPhenols', 0.1, 0.7, 0.3)
        proanthocyanins = st.sidebar.slider('Proanthocyanins', 0.4, 3.6, 1.5)
        color = st.sidebar.slider('Color', 1.2, 13.0, 5.0)
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

    input_df = get_user_inputs()

    # --- MAIN PANEL: DATASET PREVIEW ---
    st.subheader(f"Dataset Preview: First {num_rows} Rows")
    st.dataframe(df.head(num_rows))

    # --- MAIN PANEL: PREDICTION ---
    st.subheader("Your Input Parameters")
    st.write(input_df)

    # Perform prediction for Barolo, Grignolino, or Barbera [2, 3]
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    st.success(f"The predicted wine class is: **{prediction.upper()}**")

    # Bar graph for confidence visualization
    st.subheader("Prediction Probability Bar Graph")
    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
    st.bar_chart(proba_df.T)

except FileNotFoundError:
    st.error("Dataset Error: 'wine.csv' was not found. Please upload it to your app folder.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
