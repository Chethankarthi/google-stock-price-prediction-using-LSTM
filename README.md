This project aims to predict the future closing prices of Google stocks using Long Short-Term Memory (LSTM) neural networks, a type of recurrent neural network (RNN) particularly effective for time series forecasting.

Dataset
The dataset used for this project consists of historical daily stock prices of Google. It includes the following features:

Date: The date of the trading day
Open: The opening price of the stock on the trading day
High: The highest price of the stock on the trading day
Low: The lowest price of the stock on the trading day
Close: The closing price of the stock on the trading day
Volume: The volume of stocks traded on the trading day
Approach
Data Preprocessing:
The dataset is preprocessed to handle missing values, convert data types, and normalize the numerical features.
Model Building:
An LSTM model is constructed using PyTorch, a popular deep learning framework.
The model takes input sequences of historical stock prices and predicts the future closing price.
Hyperparameters such as the number of LSTM layers, hidden units, and learning rate are tuned to optimize performance.
Training:
The LSTM model is trained on historical stock price data.
Training involves feeding batches of input sequences to the model and adjusting the model parameters to minimize the prediction error.
Evaluation:
The trained model's performance is evaluated using metrics such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE).
Performance metrics are compared against baseline models or industry standards to assess the model's effectiveness.
Prediction:
Once trained, the LSTM model is used to make predictions on unseen data.
Predictions are made for future closing prices of Google stocks, enabling stakeholders to make informed decisions.


Requirements
Python 3.x
PyTorch
pandas
numpy
scikit-learn
plotly



Results
The LSTM model achieves [Training MAE: 0.0425
Training MSE: 0.0061
Training RMSE: 0.0781
Testing MAE: 0.0304
Testing MSE: 0.0016
Testing RMSE: 0.0397] on the test dataset, indicating a accurate LSTM model for the google stock price prediction.



CHETHAN B
Data Science BATCH MDE85