import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(page_title="Wine Class Predictor", layout="wide")

st.title("Wine Class Prediction App")
st.write("""
This app predicts the **Wine Class** based on chemical composition 
and allows you to preview the source dataset.
""")

# 1. Load data and train the model
@st.cache_data
def load_data_and_model():
    # Load dataset based on headers in the sources [2]
    df = pd.read_csv("wine.csv")
    
    # Cleaning: Remove rows with missing data (indicated by trailing commas in sources [2, 5])
    df = df.dropna()
    
    X = df.drop('WineClass', axis=1)
    y = df['WineClass']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return df, model

try:
    df, model = load_data_and_model()

    # --- SIDEBAR: DATASET PREVIEW SETTINGS ---
    st.sidebar.header("Dataset Preview Settings")
    # User choice for viewing 5, 8, or more rows
    num_rows = st.sidebar.number_input("Select rows to preview:", min_value=1, max_value=len(df), value=5)

    # --- SIDEBAR: USER INPUTS ---
    st.sidebar.header("Input Wine Features")

    def get_user_inputs():
        # Ranges are based on data points in the sources (e.g., Mg up to 162, Proline up to 1680) [5-7]
        alcohol = st.sidebar.slider('Alcohol', 11.0, 15.0, 13.0)
        malic_acid = st.sidebar.slider('MalicAcid', 0.7, 6.0, 2.3)
        ash = st.sidebar.slider('Ash', 1.3, 3.5, 2.3)
        alkalinity_ash = st.sidebar.slider('AlkalinityAsh', 10.0, 30.0, 19.0)
        mg = st.sidebar.slider('Mg (Magnesium)', 70, 170, 100)
        phenols = st.sidebar.slider('Phenols', 0.9, 4.0, 2.3)
        flavanoids = st.sidebar.slider('Flavanoids', 0.3, 5.2, 2.0)
        non_flavanoid_phenols = st.sidebar.slider('NonFlavanoidPhenols', 0.1, 0.7, 0.3)
        proanthocyanins = st.sidebar.slider('Proanthocyanins', 0.4, 3.6, 1.5)
        color = st.sidebar.slider('Color Intensity', 1.2, 13.0, 5.0)
        hue = st.sidebar.slider('Hue', 0.4, 1.8, 1.0)
        od_ratio = st.sidebar.slider('ODRatio', 1.2, 4.0, 2.6)
        proline = st.sidebar.slider('Proline', 270, 1700, 750)
        
        data = {
            'Alcohol': alcohol, 'MalicAcid': malic_acid, 'Ash': ash,
            'AlkalinityAsh': alkalinity_ash, 'Mg': mg, 'Phenols': phenols,
            'Flavanoids': flavanoids, 'NonFlavanoidPhenols': non_flavanoid_phenols,
            'Proanthocyanins': proanthocyanins, 'Color': color, 'Hue': hue,
            'ODRatio': od_ratio, 'Proline': proline
        }
        # FIX: The index= below resolves the SyntaxError from your screenshot [1]
        return pd.DataFrame(data, index=)

    input_df = get_user_inputs()

    # --- MAIN PANEL: DATASET PREVIEW ---
    st.subheader(f"Dataset Preview: First {num_rows} Rows")
    st.dataframe(df.head(num_rows))

    # --- MAIN PANEL: PREDICTION ---
    st.subheader("Your Input Parameters")
    st.write(input_df)

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    # Result will be barolo, grignolino, or barbera [2-4]
    st.success(f"The predicted wine class is: **{prediction.upper()}**")

    # Bar graph for probability [as requested]
    st.subheader("Prediction Probability Bar Graph")
    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
    st.bar_chart(proba_df.T)

except FileNotFoundError:
    st.error("Error: 'wine.csv' not found. Please ensure the file is in the same directory.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
