import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(page_title="Wine Predictor", layout="wide")

st.title("Wine Class Prediction App")
st.write("Predict if a wine is Barolo, Grignolino, or Barbera based on chemical data.")

# 1. Load data and train model
@st.cache_data
def load_and_train():
    # Load dataset using headers from the source [2]
    df = pd.read_csv("wine.csv")
    
    # IMPORTANT: Your wine.csv has some broken/incomplete lines [2, 5, 6]
    # dropna() removes those messy rows so the model doesn't crash.
    df = df.dropna()
    
    X = df.drop('WineClass', axis=1)
    y = df['WineClass']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return df, model

try:
    df, model = load_and_train()

    # --- SIDEBAR: DATA PREVIEW ---
    st.sidebar.header("Dataset Preview")
    num_rows = st.sidebar.number_input("Rows to show:", min_value=1, max_value=len(df), value=5)

    # --- SIDEBAR: USER INPUTS ---
    st.sidebar.header("Input Features")
    
    def get_inputs():
        # Slider ranges reflect the data points in the sources [5, 7, 8]
        alcohol = st.sidebar.slider('Alcohol', 11.0, 15.0, 13.0)
        malic = st.sidebar.slider('MalicAcid', 0.7, 6.0, 2.3)
        ash = st.sidebar.slider('Ash', 1.3, 3.5, 2.3)
        alk = st.sidebar.slider('AlkalinityAsh', 10.0, 30.0, 19.0)
        mg = st.sidebar.slider('Mg', 70, 170, 100)
        phenols = st.sidebar.slider('Phenols', 0.9, 4.0, 2.3)
        flav = st.sidebar.slider('Flavanoids', 0.3, 5.2, 2.0)
        non_flav = st.sidebar.slider('NonFlavanoidPhenols', 0.1, 0.7, 0.3)
        proanth = st.sidebar.slider('Proanthocyanins', 0.4, 3.6, 1.5)
        color = st.sidebar.slider('Color', 1.2, 13.0, 5.0)
        hue = st.sidebar.slider('Hue', 0.4, 1.8, 1.0)
        od = st.sidebar.slider('ODRatio', 1.2, 4.0, 2.6)
        proline = st.sidebar.slider('Proline', 270, 1700, 750)
        
        data = {
            'Alcohol': alcohol, 'MalicAcid': malic, 'Ash': ash,
            'AlkalinityAsh': alk, 'Mg': mg, 'Phenols': phenols,
            'Flavanoids': flav, 'NonFlavanoidPhenols': non_flav,
            'Proanthocyanins': proanth, 'Color': color, 'Hue': hue,
            'ODRatio': od, 'Proline': proline
        }
        # FIXED: Added  to the index to resolve your SyntaxError [1]
        return pd.DataFrame(data, index=)

    user_data = get_inputs()

    # --- MAIN PANEL ---
    st.subheader(f"Top {num_rows} Rows of Dataset")
    st.dataframe(df.head(num_rows))

    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data)

    st.subheader("Prediction")
    st.success(f"Predicted Class: **{prediction.upper()}**")

    st.subheader("Probability Bar Graph")
    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
    st.bar_chart(proba_df.T)

except FileNotFoundError:
    st.error("Missing 'wine.csv' file in repository.")
except Exception as e:
    st.error(f"Error: {e}")
