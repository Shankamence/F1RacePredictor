import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv('data/f1_race_results_2024_rounds_1_to_15.csv')

# Exclude round 2 from the data
df = df[df['round'] != 2]

# Feature Engineering
df['positionChange'] = df['gridPosition'] - df['finishPosition']
df['avgPointsPerRace'] = df.groupby('driverId')['points'].transform('mean')
df['cumulativePoints'] = df.groupby('driverId')['points'].cumsum()
df['avgFinishPosition'] = df.groupby('driverId')['finishPosition'].transform('mean')
df['driverId_encoded'] = df.groupby('driverId')['finishPosition'].transform('mean')
df['constructorId_encoded'] = df.groupby('constructorId')['finishPosition'].transform('mean')

# Convert 'fastestLapTime' to total seconds
def convert_to_seconds(time_str):
    if pd.isnull(time_str):
        return np.nan
    mins, secs = time_str.split(':')
    return int(mins) * 60 + float(secs)

df['fastestLapTime'] = df['fastestLapTime'].apply(convert_to_seconds)

# Convert 'raceTime' to total seconds
def convert_race_time(time_str):
    if pd.isnull(time_str):
        return np.nan
    time_parts = time_str.split(':')
    if len(time_parts) == 3:  # h:mm:ss.sss
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = float(time_parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(time_parts) == 2:  # m:ss.sss
        minutes = int(time_parts[0])
        seconds = float(time_parts[1])
        return minutes * 60 + seconds
    return np.nan

df['raceTime'] = df['raceTime'].apply(convert_race_time)

# Use Pandas get_dummies as an alternative to OneHotEncoder
encoded_df = pd.get_dummies(df[['constructorId', 'driverId']], drop_first=True)

# Select features including engineered features
numerical_features = df[['gridPosition', 'laps', 'raceTimeMillis', 'fastestLapTime',
                         'positionChange', 'avgPointsPerRace', 'cumulativePoints',
                         'avgFinishPosition', 'driverId_encoded', 'constructorId_encoded']]

# Combine encoded features with numerical features
features = np.hstack((numerical_features, encoded_df))

# Define the target variable: Predicting the finish position
target = df['finishPosition']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)


# Fill NaN values in 'raceTimeMillis' with the mean of the column
X_train[:, 2] = np.nan_to_num(X_train[:, 2], nan=np.nanmean(X_train[:, 2]))
X_test[:, 2] = np.nan_to_num(X_test[:, 2], nan=np.nanmean(X_test[:, 2]))

# Fill NaN values in 'fastestLapTime' with the mean of the column
X_train[:, 3] = np.nan_to_num(X_train[:, 3], nan=np.nanmean(X_train[:, 3]))
X_test[:, 3] = np.nan_to_num(X_test[:, 3], nan=np.nanmean(X_test[:, 3]))

# After filling NaN values, you can proceed with model training


model.fit(X_train, y_train)

# Predict the target for the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")



