Overview
This project predicts the opening prices of ES Futures (E-mini S&P 500 Futures) using a Recurrent Neural Network (RNN) built with Long Short-Term Memory (LSTM) layers. The model leverages historical price data to forecast the next day's opening price and evaluates its performance using Root Mean Squared Error (RMSE).

Features
Data Preprocessing: Normalizes the data and prepares sequences for training.
LSTM-based RNN: Uses three LSTM layers with dropout regularization.
Training & Validation: Implements early stopping to optimize training.
Prediction: Rescales predictions back to original price values for evaluation.
Visualization: Plots true vs. predicted prices.
Export Results: Saves predictions, training history, and the best-performing model.
Prerequisites
Install the following Python libraries before running the script:

numpy
pandas
json
joblib
scikit-learn
matplotlib
tensorflow

You can install these libraries using:
pip install numpy pandas joblib scikit-learn matplotlib tensorflow

Files and Structure
es_futures.csv: Input CSV file containing historical ES Futures data. Ensure it follows the required structure:
Skip the first two rows (headers).
Columns: ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'].
best_es_rnn_model.keras: Saved model file for future predictions.
es_futures_predictions.csv: CSV file containing true and predicted prices.
training_history.json: JSON file storing training performance metrics.

How It Works
Load Data: Reads historical ES Futures data and sorts it by date.
Preprocessing:
Normalizes the opening prices to a range of 0 to 1.
Creates sequences of length 60 for training.
Model Training:
Defines an LSTM-based RNN with multiple layers.
Trains the model using early stopping to avoid overfitting.
Evaluation:
Predicts prices on the test set and rescales them to the original scale.
Calculates RMSE to evaluate prediction accuracy.
Visualization:
Plots the true prices vs. predicted prices.
Save Results:
Saves the best model, predictions, and training history.

Usage
Update File Directory:
Modify the script to set the correct working directory:
os.chdir('your_file_directory')

Prepare Data:
Ensure the es_futures.csv file is in the correct format and location.
Run the Script:
Execute the script:
python your_script_name.py

View Results:
The predictions and training history are saved as CSV and JSON files.
The model is saved as best_es_rnn_model.keras.

Results
RMSE: The Root Mean Squared Error is printed after evaluation.
Next Day Prediction: The predicted opening price for the next day is displayed.
Visualization: A graph compares the true and predicted prices.

Customization
Sequence Length: Modify the sequence_length variable to adjust how many previous days the model considers.
Model Architecture: Add or remove LSTM layers, or tweak neurons and dropout rates.
Batch Size: Update the batch_sizes array to test different configurations.
Prediction Target: Adjust the y variable to predict other price metrics.

License
This project is open-source and can be freely modified and distributed.
