import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

st.title("Wine Class Prediction App")
st.write("This app predicts the wine class (Barolo, Grignolino, or Barbera) based on chemical properties.")

# Load the dataset to train the model (as discussed in our history)
@st.cache_data
def load_and_train():
    # The source data indicates headers: WineClass, Alcohol, MalicAcid, etc. [1]
    df = pd.read_csv("wine.csv")
    X = df.drop('WineClass', axis=1)
    y = df['WineClass']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

try:
    model = load_and_train()

    st.sidebar.header("Input Wine Features")

    def user_input_features():
        # Ranges below are based on data points found in sources [1-4]
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
        # FIXED: Added  to the index to prevent the previous syntax error
        return pd.DataFrame(data, index=)

    input_df = user_input_features()

    st.subheader("User Input Parameters")
    st.write(input_df)

    # Prediction based on classes: barolo, grignolino, barbera [1, 5, 6]
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction")
    st.write(f"The predicted wine class is: **{prediction}**")

    st.subheader("Prediction Probability")
    st.write(pd.DataFrame(prediction_proba, columns=model.classes_))

except FileNotFoundError:
    st.error("Error: 'wine.csv' not found. Please ensure the dataset is in the same folder.")
except Exception as e:
    st.error(f"An error occurred: {e}")
