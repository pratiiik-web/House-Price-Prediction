import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def train_model():
    df = pd.read_csv('Data/Housing.csv')

    binary_cols = ['mainroad', 'guestroom', 'basement',
                   'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    df['furnishingstatus'] = df['furnishingstatus'].map({
        'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0
    })

    df['price_per_sqft'] = df['price'] / df['area']
    df['bath_per_bed']   = df['bathrooms'] / (df['bedrooms'] + 1)
    df['total_rooms']    = df['bedrooms'] + df['bathrooms']

    X = df.drop('price', axis=1)
    y = df['price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = train_model()

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.markdown("Fill in the details to get an estimated house price.")

col1, col2 = st.columns(2)

with col1:
    area       = st.number_input("Area (sq ft)", 500, 20000, 5000)
    bedrooms   = st.slider("Bedrooms", 1, 6, 3)
    bathrooms  = st.slider("Bathrooms", 1, 4, 1)
    stories    = st.slider("Stories", 1, 4, 1)
    parking    = st.slider("Parking spots", 0, 3, 1)
    furnishing = st.selectbox("Furnishing", ["furnished", "semi-furnished", "unfurnished"])

with col2:
    mainroad  = st.selectbox("Main Road access", ["yes", "no"])
    guestroom = st.selectbox("Guest Room", ["yes", "no"])
    basement  = st.selectbox("Basement", ["yes", "no"])
    hotwater  = st.selectbox("Hot Water Heating", ["yes", "no"])
    aircon    = st.selectbox("Air Conditioning", ["yes", "no"])
    prefarea  = st.selectbox("Preferred Area", ["yes", "no"])

if st.button("Predict Price"):
    furnishing_map = {'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}
    yn = lambda x: 1 if x == "yes" else 0

    price_per_sqft = 5000
    bath_per_bed   = bathrooms / (bedrooms + 1)
    total_rooms    = bedrooms + bathrooms

    input_dict = {
        'area':             area,
        'bedrooms':         bedrooms,
        'bathrooms':        bathrooms,
        'stories':          stories,
        'mainroad':         yn(mainroad),
        'guestroom':        yn(guestroom),
        'basement':         yn(basement),
        'hotwaterheating':  yn(hotwater),
        'airconditioning':  yn(aircon),
        'parking':          parking,
        'prefarea':         yn(prefarea),
        'furnishingstatus': furnishing_map[furnishing],
        'price_per_sqft':   price_per_sqft,
        'bath_per_bed':     bath_per_bed,
        'total_rooms':      total_rooms,
    }

    input_df  = pd.DataFrame([input_dict])
    input_sc  = scaler.transform(input_df)
    prediction = model.predict(input_sc)[0]

    st.divider()
    st.metric("Estimated Price", f"₹{prediction:,.0f}")

    if prediction < 4000000:
        st.success("🟢 Budget friendly property")
    elif prediction < 7000000:
        st.warning("🟡 Mid-range property")
    else:
        st.error("🔴 Premium property")