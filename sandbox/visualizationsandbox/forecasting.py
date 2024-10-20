from prophet import Prophet
import pandas as pd

df = pd.read_pickle('../final_draft/df.pkl')

df['ds'] = pd.to_datetime(df['DATETIME_DTM'], format='%Y-%m-%d')
df.rename({'ovr_danger': 'y'}, axis=1, inplace=True)
df = df[['ds', 'y']]


m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=30)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
