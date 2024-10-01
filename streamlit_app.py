import joblib
from PIL import Image
import pandas as pd
import streamlit as st
from datetime import date

# loading pre-trained regressor
model = joblib.load(open('regressor.sav', 'rb'))

# loading pre-trained OHE
with open('encoder.sav', 'rb') as f:
    enc = joblib.load(f)

st.set_page_config(
        page_title="Car price predictor!!",
        page_icon="🏎",
)

# main App layout
c30, c31, c32 = st.columns([4, 1, 3])

with c30:
    st.title("🏎 Car price predictor")
    st.header("")

with st.expander("ℹ️ - About this app", expanded=True):

        st.write(
        """     
        -   The *Car price predictor* app is an easy-to-use interface built with Streamlit for fast&easy fair car price assessment
        -   It uses widgets to get car info and Random Forest-based ML model to tell you the fair car price
        """)

        st.markdown("")
st.markdown("## ⚙️ Enter car details below")

with st.form(key="my_form"):

        ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])

        make_year = st.slider(
            "Year of production",
            min_value=1985,
            max_value=date.today().year,
            value=2015,
            help="Please select the year of car production",
        )
        car_age = int(date.today().year-make_year)

        mileage = st.slider(
            "Car mileage(km)",
            min_value=1,
            max_value=499999,
            value=9000,
            help="Please select car mileage"
        )
        
        fuel_type = st.selectbox(
        "Fuel type:", ['Diesel', 'Petrol'], index=0)

        seller_type = st.selectbox(
        "Seller type:", ['Dealer', 'Individual'], index=0)

        transmission = st.selectbox(
        "Transmission type:", ['Manual', 'Automatic'], index=0)

        prev_owners = st.number_input(
            "Number of previous owners",
            value=1,
            min_value=1,
            max_value=12,
            help='Please provide the number of previous owners of the car')

        features = [[mileage, fuel_type, seller_type, transmission, prev_owners, car_age]]
        submitted = st.form_submit_button(label="✨ Get the price!")

if submitted:
        df = pd.DataFrame(features, columns=['Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Past_Owners', 'Age'])
        df_main = enc.transform(df)
        
        pred=model.predict(df_main)
        # Results
        st.write('🤖 believes that the fair price for your car is: `{}`'.format(round(pred[0], 2)))
