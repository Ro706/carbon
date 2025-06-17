import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64

# Load the trained model
model = joblib.load('carbon_footprint_model.pkl')

# Function to set background image with adjusted opacity
def set_bg_image(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: white;
        }}
        .stAlert {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 1000;
            width: 500px;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 20px;
            border-radius: 10px;
        }}
        .st-bw, .st-bx, .st-cy, .st-d4, .st-cq {{
            background-color: rgba(0, 0, 0, 0.6) !important;
            color: white !important;
        }}
        .stButton>button {{
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
set_bg_image('background.jpg')

# Define the Streamlit app
def main():
    st.title('Carbon Footprint Calculator')

    with st.sidebar:
        st.header("Personal Information")
        body_type = st.radio('Body Type', ['Slim', 'Average', 'Athletic', 'Heavy'])
        sex = st.radio('Sex', ['Male', 'Female'])

    with st.expander("Lifestyle", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            diet = st.radio('Diet', ['Vegan', 'Vegetarian', 'Non-Vegetarian'])
            shower_frequency = st.slider('Showers per Week', min_value=0, max_value=30, value=7)
            heating_energy_source = st.radio('Heating Energy Source', ['Electricity', 'Natural Gas', 'Oil', 'Renewable'])
            transport = st.radio('Primary Mode of Transport', ['Car', 'Public Transport', 'Bicycle', 'Walking'])
        with col2:
            vehicle_type = st.radio('Vehicle Type', ['Sedan', 'SUV', 'Truck', 'None'])
            social_activity = st.slider('Social Activities per Month', min_value=0, max_value=30, value=5)
            monthly_grocery_bill = st.slider('Monthly Grocery Bill (USD)', min_value=0, max_value=2000, value=200)
            air_travel_frequency = st.slider('Flights per Year', min_value=0, max_value=50, value=2)

    with st.expander("Energy and Waste", expanded=True):
        col3, col4 = st.columns(2)
        with col3:
            vehicle_monthly_distance = st.slider('Vehicle Monthly Distance (km)', min_value=0, max_value=2000, value=0)
            waste_bag_size = st.radio('Waste Bag Size', ['Small', 'Medium', 'Large'])
            waste_bag_weekly_count = st.slider('Waste Bags per Week', min_value=0, max_value=20, value=2)
            tv_pc_daily_hours = st.slider('Daily Hours on TV/PC', min_value=0, max_value=24, value=2)
        with col4:
            new_clothes_monthly = st.slider('New Clothes Purchased Monthly', min_value=0, max_value=20, value=1)
            internet_daily_hours = st.slider('Daily Hours on Internet', min_value=0, max_value=24, value=4)
            energy_efficiency = st.radio('Energy Efficiency', ['Low', 'Medium', 'High'])
            recycling = st.radio('Recycling Habits', ['Never', 'Sometimes', 'Often', 'Always'])
            cooking_with = st.radio('Cooking With', ['Electricity', 'Gas', 'Induction'])

    if st.button('Calculate Carbon Footprint'):
        # Convert categorical inputs to numerical values
        body_type_num = {'Slim': 0, 'Average': 1, 'Athletic': 2, 'Heavy': 3}[body_type]
        sex_num = {'Male': 0, 'Female': 1}[sex]
        diet_num = {'Vegan': 0, 'Vegetarian': 1, 'Non-Vegetarian': 2}[diet]
        heating_energy_source_num = {'Electricity': 0, 'Natural Gas': 1, 'Oil': 2, 'Renewable': 3}[heating_energy_source]
        transport_num = {'Car': 0, 'Public Transport': 1, 'Bicycle': 2, 'Walking': 3}[transport]
        vehicle_type_num = {'Sedan': 0, 'SUV': 1, 'Truck': 2, 'None': 3}[vehicle_type]
        waste_bag_size_num = {'Small': 0, 'Medium': 1, 'Large': 2}[waste_bag_size]
        energy_efficiency_num = {'Low': 0, 'Medium': 1, 'High': 2}[energy_efficiency]
        recycling_num = {'Never': 0, 'Sometimes': 1, 'Often': 2, 'Always': 3}[recycling]
        cooking_with_num = {'Electricity': 0, 'Gas': 1, 'Induction': 2}[cooking_with]

        # Prepare input data for prediction
        input_data = np.array([[body_type_num, sex_num, diet_num, shower_frequency, heating_energy_source_num,
                                transport_num, vehicle_type_num, social_activity, monthly_grocery_bill,
                                air_travel_frequency, vehicle_monthly_distance, waste_bag_size_num,
                                waste_bag_weekly_count, tv_pc_daily_hours, new_clothes_monthly,
                                internet_daily_hours, energy_efficiency_num, recycling_num, cooking_with_num]])

        # Predict carbon footprint
        carbon_footprint = model.predict(input_data)[0]

         # Calculate the number of trees to plant per month
        trees_per_month = (carbon_footprint / 21.77) / 12

        # Display results in a popup-like alert
        st.markdown(
            f"""
            <div style='background-color: rgba(0, 0, 0, 0.8); color: white; padding: 20px; border-radius: 10px;'>
                <h4>Carbon Footprint Result</h4>
                <p>Your estimated carbon footprint is: <strong>{carbon_footprint:.2f} kg CO2/year</strong>.</p>
                <p>You need to plant approximately <strong>{trees_per_month:.0f} trees per month</strong> to neutralize your carbon footprint.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == '__main__':
    main()
