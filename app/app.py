import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Get path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")
DATA_DIR = os.path.join(BASE_DIR, "../data")

# Load models
reg_model = joblib.load(os.path.join(MODELS_DIR, "regressor.pkl"))
clf_model = joblib.load(os.path.join(MODELS_DIR, "classifier.pkl"))
encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
feature_cols = joblib.load(os.path.join(MODELS_DIR, "features.pkl"))

st.title("🏡 Real Estate Investment Advisor")

# Sidebar
st.sidebar.header("Enter Property Details")

data = {
    'State': st.sidebar.text_input("State", "Telangana"),
    'City': st.sidebar.text_input("City", "Hyderabad"),
    'Locality': st.sidebar.text_input("Locality", "Gachibowli"),
    'Property_Type': st.sidebar.selectbox("Type", ["Apartment","Villa","House"]),
    'BHK': st.sidebar.slider("BHK",1,5,2),
    'Size_in_SqFt': st.sidebar.number_input("Size",500,5000,1200),
    'Price_in_Lakhs': st.sidebar.number_input("Price",10,500,75),
    'Year_Built': st.sidebar.number_input("Year",1990,2025,2015),
    'Furnished_Status': st.sidebar.selectbox("Furnished",["Unfurnished","Semi","Fully"]),
    'Floor_No': st.sidebar.number_input("Floor",0,50,2),
    'Total_Floors': st.sidebar.number_input("Total Floors",1,100,10),
    'Nearby_Schools': st.sidebar.slider("Schools",0,10,3),
    'Nearby_Hospitals': st.sidebar.slider("Hospitals",0,10,2),
    'Public_Transport_Accessibility': st.sidebar.slider("Transport",0,10,5),
    'Parking_Space': st.sidebar.slider("Parking",0,5,1),
    'Security': st.sidebar.selectbox("Security",["Gated","CCTV","Guard"]),
    'Amenities': st.sidebar.selectbox("Amenities",["Basic","Gym","Pool","Clubhouse"]),
    'Facing': st.sidebar.selectbox("Facing",["North","South","East","West"]),
    'Owner_Type': st.sidebar.selectbox("Owner",["Individual","Builder","Agent"]),
    'Availability_Status': st.sidebar.selectbox("Status",["Available","Under Construction","Sold"])
}

input_df = pd.DataFrame([data])

# Preprocess
def preprocess(df):
    df['Price_per_SqFt'] = df['Price_in_Lakhs'] * 100000 / df['Size_in_SqFt']
    df['Age_of_Property'] = 2025 - df['Year_Built']

    for col, le in encoders.items():
        try:
            df[col] = le.transform(df[col].astype(str))
        except:
            df[col] = 0

    return df

processed = preprocess(input_df)
processed = processed.reindex(columns=feature_cols, fill_value=0)

# Prediction
if st.button("Predict"):

    price_log = reg_model.predict(processed)[0]
    price = np.expm1(price_log)
    invest = clf_model.predict(processed)[0]

    st.subheader("Results")

    if invest == 1:
        st.success("✅ Good Investment")
    else:
        st.error("❌ Not Good Investment")

    st.info(f"💰 Future Price: ₹ {round(price,2)} Lakhs")

    prob = clf_model.predict_proba(processed)[0][1]
    st.write(f"Confidence: {round(prob*100,2)}%")

# Feature Importance
st.subheader("📈 Feature Importance")

fig, ax = plt.subplots()
ax.barh(feature_cols, reg_model.feature_importances_)
st.pyplot(fig)

# Visualizations
st.subheader("📊 Data Insights")

df = pd.read_csv(os.path.join(DATA_DIR, "india_housing_prices.csv"))

fig1, ax1 = plt.subplots()
df['Price_in_Lakhs'].hist(ax=ax1)
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
df['BHK'].value_counts().plot(kind='bar', ax=ax2)
st.pyplot(fig2)

# Map
st.subheader("📍 Location Map")

map_data = pd.DataFrame({'lat':[17.3850],'lon':[78.4867]})
st.map(map_data)