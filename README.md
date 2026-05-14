# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset
Weather prediction is important for analyzing atmospheric conditions and forecasting future weather changes.
This project uses the Random Forest Algorithm to predict weather conditions based on environmental features such as temperature, humidity, and wind speed.


## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and Prepare the Dataset
Import the weather dataset, select environmental features as input variables (X) and weather condition as the target variable (y), then split the data into training and testing sets.
Initialize the Random Forest Model
Create the Random Forest Classifier model and set parameters such as number of trees (n_estimators) and random state.
Train the Model
Fit the Random Forest model using the training dataset so that multiple decision trees are created for weather prediction.
Predict and Evaluate the Model
Predict weather conditions for test data and evaluate the model performance using accuracy score, confusion matrix, and classification report. 

## Program:
```
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()
df['time'] = pd.to_datetime(df['time'], errors='coerce')

print("Original rows:", len(df))

# Only drop if target missing
df = df.dropna(subset=['tem', 'pm2_5'])

# Fill feature columns instead of dropping
df['hum'] = df['hum'].fillna(df['hum'].mean())
df['pressure'] = df['pressure'].fillna(df['pressure'].mean())
df['wind_speed'] = df['wind_speed'].fillna(df['wind_speed'].mean())
df['co2'] = df['co2'].fillna(df['co2'].mean())

# Sort by time
df = df.sort_values('time')

# Create lag features
df['Temp_Lag1'] = df['tem'].shift(1)
df['PM_Lag1'] = df['pm2_5'].shift(1)

# Only remove first row created by shift
df = df.iloc[1:]

print("Rows after preprocessing:", len(df))

# Features
X = df[['hum', 'pressure', 'wind_speed', 'co2',
        'Temp_Lag1', 'PM_Lag1']]

y_temp = df['tem']
y_pm = df['pm2_5']

print("Training samples:", len(X))

# Train models
model_temp = RandomForestRegressor(n_estimators=300, random_state=42)
model_pm = RandomForestRegressor(n_estimators=300, random_state=42)

model_temp.fit(X, y_temp)
model_pm.fit(X, y_pm)

# Save models
joblib.dump(model_temp, "temperature_model.pkl")
joblib.dump(model_pm, "pm25_model.pkl")

print("Models trained and saved successfully!")

/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Akshaya Sree G
RegisterNumber: 212225230011 
*/
```

## Output:

<img width="1263" height="463" alt="image" src="https://github.com/user-attachments/assets/8780c6c6-f583-4256-b263-62c35286fe87" />
<img width="1268" height="460" alt="image" src="https://github.com/user-attachments/assets/547534ca-5a00-4352-9ad3-0bc8d07a8720" />
<img width="1271" height="465" alt="image" src="https://github.com/user-attachments/assets/a354d3d5-dc21-4f9c-9e87-932541b494b5" />
<img width="1246" height="96" alt="image" src="https://github.com/user-attachments/assets/246b0448-ed99-4fcc-823a-3dcd81bce80f" />



## Result:
The Random Forest model successfully predicted temperature, PM2.5 pollution, and solar radiation using weather sensor data with good accuracy. The system also generated next-step predictions and visual graphs comparing actual vs predicted values and showing feature importance.
