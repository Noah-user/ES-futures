import numpy as np
import pandas as pd
import json
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

import os
os.chdir('your_file_directory')

data = pd.read_csv('es_futures.csv', skiprows=2, index_col=0, parse_dates=True)
data.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
data = data.sort_index(ascending=True)
opening_prices = data[['Open']]

# normalize data between 0 and 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(opening_prices)

# Create sequences and labels can adjust sequence
sequence_length = 60

X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])  # The next day's opening price
X, y = np.array(X), np.array(y)

# Reshape X for LSTM input
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the RNN model with additional neurons, layers, and Dropout
model = Sequential([
    LSTM(200, return_sequences=True, input_shape=(X_train.shape[1], 1)),  # First LSTM layer
    Dropout(0.2),
    LSTM(100, return_sequences=True),  # Second LSTM layer
    Dropout(0.2),
    LSTM(50, return_sequences=False),  # Third LSTM layer
    Dense(1)  # Output layer for predicting next day's price
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

batch_sizes = [16] # can change or add multiple batch_sizes
for batch_size in batch_sizes:
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )

# Make predictions on the test set
predictions = model.predict(X_test)

# Rescale the predictions back to the original scale
predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))


# Calculate RMSE to determine errors
rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# plot
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='True Prices', color='blue')
plt.plot(predictions_rescaled, label='Predicted Prices', color='red')
plt.legend()
plt.title('ES Futures Price Prediction: True vs Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

# Save the best model
model.save('best_es_rnn_model.keras')
print("Best model saved as 'best_es_rnn_model.keras'")

# Predict the next day's price
recent_sequence = scaled_data[-sequence_length:]
recent_sequence = recent_sequence.reshape(1, sequence_length, 1)
next_day_prediction = model.predict(recent_sequence)
next_day_price = scaler.inverse_transform(next_day_prediction)[0, 0]
print(f"Predicted next day's opening price: {next_day_price}")

# Create DataFrame with true and predicted prices
results_df = pd.DataFrame({
    'True Prices': y_test_rescaled.flatten(),
    'Predicted Prices': predictions_rescaled.flatten()
})

# Save to CSV
results_df.to_csv('es_futures_predictions.csv', index=False)
print("Predictions saved to 'es_futures_predictions.csv'")

# Save training history
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)
print("Training history saved to 'training_history.json'")

exit()