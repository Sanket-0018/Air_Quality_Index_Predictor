import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Skip first 4 rows — they contain metadata, not actual data
df = pd.read_csv("Pune_Air_Pollutants.csv", skiprows=4)

# Show the first few rows

df.columns = ['Datetime', 'PM2.5', 'PM10', 'PM1', 'O3', 'CO', 'NO', 'NO2', 'NOx', 'CO2']
print(df.head())
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(df.isnull().sum())
df['Tomorrow_PM2.5'] = df['PM2.5'].shift(-1)
df = df.dropna()
print(df.head())


# Load the dataset (skipping metadata rows)
df = pd.read_csv("Pune_Air_Pollutants.csv", skiprows=4)

# Rename the columns for clarity (optional if already renamed)
df.columns = ['Timestamp', 'PM2.5', 'PM1', 'O3', 'CO', 'NO', 'NO2', 'NOx', 'CO2','extra']

# Drop rows with missing values in relevant columns
df = df[['PM2.5', 'PM1', 'NO2', 'CO']].dropna()

# Features and target
X = df[['PM1', 'NO2', 'CO']]
y = df[['PM2.5']]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = knn.predict(X_test_scaled)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# --- User Prediction Part ---
print("\n--- Predict Tomorrow's AQI (PM2.5) ---")
pm1 = float(input("Enter current PM1: "))
no2 = float(input("Enter current NO2: "))
co = float(input("Enter current CO: "))

# Prepare and scale user input
user_input = [[pm1, no2, co]]
user_input_scaled = scaler.transform(user_input)

# Predict tomorrow's PM2.5
prediction = knn.predict(user_input_scaled)

# Prediction
prediction = knn.predict(user_input_scaled)
predicted_pm25 = round(prediction[0][0], 2)

# Category labeling
def get_aqi_category(predicted_pm25):
    if predicted_pm25 <= 12:
        return "Good"
    elif predicted_pm25 <= 35:
        return "Moderate"
    elif predicted_pm25 <= 55:
        return "Unhealthy for Sensitive Groups"
    elif predicted_pm25 <= 150:
        return "Unhealthy"
    elif predicted_pm25 <= 250:
        return "Very Unhealthy"
    else:
        return "Hazardous"

category = get_aqi_category(predicted_pm25)

# Display result
print(f"Predicted PM2.5 (Tomorrow's AQI): {predicted_pm25}")
print(f"Air Quality Category: {category}")
