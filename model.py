import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
data = pd.read_csv('carbon_footprint_data.csv')

# Preprocess categorical data
label_encoders = {}
categorical_columns = data.select_dtypes(include=['object']).columns

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column].astype(str))

# Separate features and target variable
X = data.drop('CarbonEmission', axis=1)
y = data['CarbonEmission']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

# Summary Printout
print("\n--- Model Performance Summary ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.2f}")
print(f"R-squared Score (R²): {r2:.4f}")
print(f"Model Accuracy (based on R²): {r2 * 100:.2f}%")
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean CV R² Score: {np.mean(cv_scores):.4f}\n")

# Feature Importance
feature_importances = pd.DataFrame(model.feature_importances_,
                                   index=X.columns,
                                   columns=['Importance']).sort_values('Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.Importance, y=feature_importances.index)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Carbon Emission')
plt.ylabel('Predicted Carbon Emission')
plt.title('Actual vs Predicted Carbon Emission')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()

# Save the trained model
joblib.dump(model, 'carbon_footprint_model.pkl')

print("✅ Model training complete and saved as 'carbon_footprint_model.pkl'.")
print("📊 Plots saved: 'feature_importance.png', 'actual_vs_predicted.png'")
