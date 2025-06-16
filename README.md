# Carbon Footprint Calculator

This project provides a Streamlit web application and a machine learning model to estimate and visualize your annual carbon footprint based on your lifestyle and consumption habits.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [App Details](#app-details)
- [Screenshots](#screenshots)
- [License](#license)

---

## Features

- Interactive Streamlit UI for user input
- Predicts annual carbon emissions (kg CO₂/year)
- Visualizes feature importance and prediction accuracy
- Customizable background and UI styling
- Model built with Random Forest Regressor

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/carbon-footprint-calculator.git
    cd carbon-footprint-calculator
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the dataset:**
    - Place your `carbon_footprint_data.csv` in the project directory.

---

## Model Training

The model is trained using a Random Forest Regressor on your lifestyle and consumption data.

**Training Script:**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

# Load dataset
data = pd.read_csv('carbon_footprint_data.csv')

# Encode categorical columns
label_encoders = {}
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
     label_encoders[column] = LabelEncoder()
     data[column] = label_encoders[column].fit_transform(data[column].astype(str))

# Split features and target
X = data.drop('CarbonEmission', axis=1)
y = data['CarbonEmission']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("CV Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Feature importance plot
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.importance, y=feature_importances.index)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Actual vs Predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Carbon Emission')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()

# Save model
joblib.dump(model, 'carbon_footprint_model.pkl')
```

---

## App Details

The Streamlit app collects user data, processes it, and predicts your carbon footprint using the trained model.

**App Script:**

```python
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64

# Load the trained model
model = joblib.load('carbon_footprint_model.pkl')

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

set_bg_image('background.jpg')

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
                shower_frequency = st.slider('Showers per Week', 0, 30, 7)
                heating_energy_source = st.radio('Heating Energy Source', ['Electricity', 'Natural Gas', 'Oil', 'Renewable'])
                transport = st.radio('Primary Mode of Transport', ['Car', 'Public Transport', 'Bicycle', 'Walking'])
          with col2:
                vehicle_type = st.radio('Vehicle Type', ['Sedan', 'SUV', 'Truck', 'None'])
                social_activity = st.slider('Social Activities per Month', 0, 30, 5)
                monthly_grocery_bill = st.slider('Monthly Grocery Bill (USD)', 0, 2000, 200)
                air_travel_frequency = st.slider('Flights per Year', 0, 50, 2)

     with st.expander("Energy and Waste", expanded=True):
          col3, col4 = st.columns(2)
          with col3:
                vehicle_monthly_distance = st.slider('Vehicle Monthly Distance (km)', 0, 2000, 0)
                waste_bag_size = st.radio('Waste Bag Size', ['Small', 'Medium', 'Large'])
                waste_bag_weekly_count = st.slider('Waste Bags per Week', 0, 20, 2)
                tv_pc_daily_hours = st.slider('Daily Hours on TV/PC', 0, 24, 2)
          with col4:
                new_clothes_monthly = st.slider('New Clothes Purchased Monthly', 0, 20, 1)
                internet_daily_hours = st.slider('Daily Hours on Internet', 0, 24, 4)
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

          input_data = np.array([[body_type_num, sex_num, diet_num, shower_frequency, heating_energy_source_num,
                                          transport_num, vehicle_type_num, social_activity, monthly_grocery_bill,
                                          air_travel_frequency, vehicle_monthly_distance, waste_bag_size_num,
                                          waste_bag_weekly_count, tv_pc_daily_hours, new_clothes_monthly,
                                          internet_daily_hours, energy_efficiency_num, recycling_num, cooking_with_num]])

          carbon_footprint = model.predict(input_data)[0]

          st.markdown(
                f"""
                <div style='background-color: rgba(0, 0, 0, 0.8); color: white; padding: 20px; border-radius: 10px;'>
                     <h4>Carbon Footprint Result</h4>
                     <p>Your estimated carbon footprint is: <strong>{carbon_footprint:.2f} kg CO2/year</strong>.</p>
                     <p>You need to plant approximately <strong>{carbon_footprint / 21.77:.2f} trees</strong> to neutralize your carbon footprint.</p>
                </div>
                """,
                unsafe_allow_html=True
          )

if __name__ == '__main__':
     main()
```

---

## Usage

1. **Train the model** (if not already trained):
    ```bash
    python model.py
    ```
    This will generate `carbon_footprint_model.pkl`.

2. **Run the Streamlit app:**
    ```bash
    python -m streamlit run app.py
    ```

3. **Open your browser** to the provided local URL and interact with the calculator.

---

## Screenshots

- `feature_importance.png`: Shows which features most influence the prediction.
- `actual_vs_predicted.png`: Visualizes model accuracy.

---

## License

This project is licensed under the MIT License.

