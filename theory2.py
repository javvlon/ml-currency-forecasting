import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load data
bids_data = pd.read_csv("GBPUSD_Hourly_Bid_2021.01.01_2024.04.11.csv")
asks_data = pd.read_csv("GBPUSD_Hourly_Ask_2021.01.01_2024.04.11.csv")

# Merge bids and asks data based on the timestamp
merged_data = pd.merge(bids_data, asks_data, on="Time (EET)")
# Extract features (Open, High, Low, Close, Volume) from both bid and ask data
features_bid = bids_data[['Open', 'High', 'Low', 'Close', 'Volume']]
features_ask = asks_data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Combine bid and ask features
features = pd.concat([features_bid, features_ask], axis=1)

# Normalize features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Split data into input (X) and output (y)
X = scaled_features[:-1]  # Use all but the last row as input
y = scaled_features[1:, 0]  # Predict the next GBP rate (Open bid)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reshape input data for LSTM (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Predict next GBP rate
last_data_point = scaled_features[-1].reshape(1, 1, scaled_features.shape[1])
next_rate_scaled = model.predict(last_data_point)
next_prediction = scaler.inverse_transform(np.array([[0, 0, 0, next_rate_scaled[0][0], 0]]))
print("Predicted Next GBP Rate:", next_prediction)