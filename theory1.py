import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

# Assume you have a DataFrame with 'Date' and 'Close' columns
# Load historical data

data = pd.read_csv('GBPUSD_Daily_Ask_2003.05.05_2023.12.31.csv')  

# Normalize data
scaler = MinMaxScaler()
data[['Close', 'Volume']] = scaler.fit_transform(data[['Close', 'Volume']])

# Define features and target
X = data[['Close', 'Volume']].values
y = data['Close'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape input data for RNN
X_train = X_train.reshape(-1, 1, 2)
X_test = X_test.reshape(-1, 1, 2)

# Step 2: Define RNN Model
model = Sequential()
model.add(SimpleRNN(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))

# Step 3: Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train Model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Step 5: Evaluate Model
accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", accuracy)

# Step 6: Make Predictions
predictions = model.predict(X_test)

# Reshape predictions and y_test to match each other
predictions = predictions.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Inverse transform the predictions and actual values to get them back to original scale
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# Calculate accuracy metrics
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)