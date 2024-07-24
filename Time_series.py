# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet


# Read in the data
data = pd.read_csv('Lumber_Data_Weekly.csv')
data['Date'] = pd.to_datetime(data['Date'] + '-1', format='%Y-%W-%w')

# Make the target stationary
data['Price_boxcox'], lam = boxcox(data['Market_Price'])
data["Price_stationary"] = data["Price_boxcox"].diff()
data.dropna(inplace=True)

# Split train and test
train = data.iloc[:-int(len(data) * 0.1)]
test = data.iloc[-int(len(data) * 0.1):]


# Fit LSTM Model

def scale_picker(data, column1, column2):
    data_to_scale = data[[column1, column2]].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_to_scale)
    scaled_df = pd.DataFrame(scaled_data, columns=[column1, column2])
    final_data = pd.concat([data['Date'].reset_index(drop=True), scaled_df], axis=1)
    return final_data, scaler


scaled_data_train, scaler = scale_picker(train, 'mortgage_avg', 'Price_stationary')
scaled_data_test, _ = scale_picker(test, 'mortgage_avg', 'Price_stationary')


def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        date_values = pd.to_numeric(data['Date'][i:i + seq_length])
        interest_rate_values = data['mortgage_avg'].iloc[i:i + seq_length].values
        market_price_values = data['Price_stationary'].iloc[i + seq_length]
        X.append(np.column_stack((date_values, interest_rate_values)))
        y.append(market_price_values)
    return np.array(X), np.array(y)



seq_length = 8  # Number of past weeks to consider for prediction
X_train, y_train = create_sequences(scaled_data_train, seq_length)
X_test, y_test = create_sequences(scaled_data_test, seq_length)

print(X_train, '\n', y_train, '\n', X_test, '\n', y_test)

model_lstm = Sequential()
model_lstm.add(LSTM(100, return_sequences=True, input_shape=(seq_length, 2)))  # Increased neurons
model_lstm.add(LSTM(100, return_sequences=False))
model_lstm.add(Dense(50))  # Increased complexity
model_lstm.add(Dense(1))

# Compile the model
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model_lstm.fit(X_train, y_train, batch_size=1, epochs=50)

# Make predictions and evaluate
predictions = model_lstm.predict(X_test)
predictions_inverse_scaled = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 1))), axis=1))[:,
                 0]


# Reverse differencing to get actual values
last_train_value = train['Price_boxcox'].iloc[-1]
forecasts_lstm = inv_boxcox(np.cumsum(predictions_inverse_scaled) + last_train_value, lam)

print(forecasts_lstm)

# Use ACF and Partial ACF chart to determine ARIMA pdq
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=80)
plot_acf(data['Price_stationary'])
plot_pacf(data['Price_stationary'], method='ywm')
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
plt.show()

# Build AR model
selector = ar_select_order(train['Price_stationary'], 20)
model_autoreg = AutoReg(train['Price_stationary'], lags=selector.ar_lags).fit()

transformed_forecasts = list(model_autoreg.forecast(steps=len(test)))
boxcox_forecasts = []
for idx in range(len(test)):
    if idx == 0:
        boxcox_forecast = transformed_forecasts[idx] + train['Price_boxcox'].iloc[-1]
    else:
        boxcox_forecast = transformed_forecasts[idx] + boxcox_forecasts[idx - 1]

    boxcox_forecasts.append(boxcox_forecast)

forecasts_autoreg = inv_boxcox(boxcox_forecasts, lam)

# Build SARIMA model
model = ARIMA(train['Price_boxcox'], order=(8, 1, 8),
              seasonal_order=(1, 1, 1, 52)).fit()
boxcox_forecasts = model.forecast(len(test))
forecasts_SARIMA = inv_boxcox(boxcox_forecasts, lam)


def plot_func(forecast1: list[float],
              forecast2: list[float],
              forecast3: list[float],
              title: str) -> None:
    """Function to plot the forecasts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train['Date'], y=train['Market_Price'], name='Train'))
    fig.add_trace(go.Scatter(x=test['Date'], y=test['Market_Price'], name='Test'))
    fig.add_trace(go.Scatter(x=test['Date'], y=forecast1, name='LSTM'))
    fig.add_trace(go.Scatter(x=test['Date'], y=forecast2, name="SARIMA"))
    fig.add_trace(go.Scatter(x=test['Date'], y=forecast3, name='Holt Winters'))
    fig.update_layout(template="simple_white", font=dict(size=18), title_text=title,
                      width=1400, title_x=0.5, height=800, xaxis_title='Date',
                      yaxis_title='Lumber Market Price')
    return fig.show()


# Fit simple model and get forecasts
model_simple = SimpleExpSmoothing(train['Market_Price']).fit(optimized=True)
forecasts_simple = model_simple.forecast(len(test))

# Fit Holt's model and get forecasts
model_holt = Holt(train['Market_Price'], damped_trend=True).fit(optimized=True)
forecasts_holt = model_holt.forecast(len(test))

# Fit Holt Winters model and get forecasts
model_holt_winters = ExponentialSmoothing(train['Market_Price'], trend='add',
                                          seasonal='add', seasonal_periods=52) \
    .fit(optimized=True)
forecasts_holt_winters = model_holt_winters.forecast(len(test))

# Plot the forecasts
plot_func(forecasts_lstm, forecasts_SARIMA, forecasts_holt_winters, "Holt-Winters Exponential Smoothing")
