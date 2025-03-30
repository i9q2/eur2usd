
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd




data = yf.download('EURUSD=X', period='5d', interval='1h')

data_prophet = yf.download('EURUSD=X', period='5d', interval='1m')
data_prophet = data_prophet.reset_index()
data_prophet.columns = data_prophet.columns.get_level_values(0) 
data_prophet = data_prophet.rename(columns={'Datetime': 'ds', 'Close': 'y'})
data_prophet = data_prophet[['ds', 'y']]
data_prophet['ds'] = pd.to_datetime(data_prophet['ds'])
data_prophet['y'] = pd.to_numeric(data_prophet['y'])
data_prophet['ds'] = data_prophet['ds'].dt.tz_localize(None)

model = Prophet()
model.fit(data_prophet)

future = model.make_future_dataframe(periods=24, freq='h') 
forecast = model.predict(future)

plt.figure(figsize=(12, 6))

plt.plot(data.index, data['Close'], label='Фактические данные', color='blue')
plt.plot(forecast['ds'], forecast['yhat'], label='Прогноз', linestyle='--', color='orange')

plt.ylabel('EUR/USD', fontsize=12)
plt.grid()
plt.legend()

plt.savefig('output.jpg')