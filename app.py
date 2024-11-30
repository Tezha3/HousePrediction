import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Custom transformer for cluster-based standardization
class ClusterStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scalers = {}

    def fit(self, X, y=None):
        self.clusters = self.kmeans.fit_predict(X)
        for cluster in np.unique(self.clusters):
            cluster_data = X[self.clusters == cluster]
            self.scalers[cluster] = StandardScaler().fit(cluster_data)
        return self

    def transform(self, X):
        clusters = self.kmeans.predict(X)
        standardized = np.zeros_like(X)
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            standardized[mask] = self.scalers[cluster].transform(X[mask])
        return standardized


def log_transform_function(X):
    return np.log1p(X)

def fillna_furnishingstatus(x):
    return x.fillna('unfurnished')

# Load the trained pipeline and model
model_and_pipeline = joblib.load('model_and_pipeline.joblib')

# Function to predict house price
def predict_price(area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    })
    
    predicted_price = model_and_pipeline.predict(input_data)

    return (np.exp(predicted_price[0]) - 1)/1000000

# Streamlit app layout
st.title("House Price Prediction App")
st.write("Enter the following details to predict the house price:")

# Input fields for user
image_path = "house.png"  # Relative path to the image
st.image(image_path, caption="Predict Your Dream Home's Estimated Price!", use_container_width=True)


# Input fields for user
area = st.number_input("Enter area (sq ft):", min_value=0, max_value=10000, value=1500)
bedrooms = st.number_input("Enter number of bedrooms:", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Enter number of bathrooms:", min_value=1, max_value=10, value=2)
stories = st.number_input("Enter number of stories:", min_value=1, max_value=5, value=2)
mainroad = st.selectbox("Is there a main road?", ["yes", "no"])
guestroom = st.selectbox("Is there a guestroom?", ["yes", "no"])
basement = st.selectbox("Is there a basement?", ["yes", "no"])
hotwaterheating = st.selectbox("Is there hot water heating?", ["yes", "no"])
airconditioning = st.selectbox("Is there air conditioning?", ["yes", "no"])
parking = st.number_input("Enter number of parking spaces:", min_value=0, max_value=10, value=2)
prefarea = st.selectbox("Is the house in a preferred area?", ["yes", "no"])
furnishingstatus = st.selectbox("Enter furnishing status:", ["furnished", "semi-furnished", "unfurnished"])

# Display the prediction when the button is clicked
if st.button("Predict Price"):
    predicted_price = predict_price(area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus)
    st.write(f"The predicted price of the house is: {predicted_price:,.2f} million")
