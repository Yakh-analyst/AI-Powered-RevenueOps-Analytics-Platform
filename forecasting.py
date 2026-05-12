import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# =========================
# REVENUE DATA
# =========================

data = {
    'ds': [
        '2025-01-01',
        '2025-02-01',
        '2025-03-01',
        '2025-04-01',
        '2025-05-01',
        '2025-06-01'
    ],
    'y': [
        25000,
        28000,
        32000,
        35000,
        39000,
        42000
    ]
}

# =========================
# CREATE DATAFRAME
# =========================

df = pd.DataFrame(data)

df['ds'] = pd.to_datetime(df['ds'])

# =========================
# TRAIN AI MODEL
# =========================

model = Prophet()

model.fit(df)

# =========================
# FUTURE PREDICTIONS
# =========================

future = model.make_future_dataframe(
    periods=6,
    freq='ME'
)

forecast = model.predict(future)

# =========================
# SHOW RESULTS
# =========================

print(
    forecast[
        ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    ].tail(6)
)

# =========================
# EXPORT CSV
# =========================

forecast.to_csv(
    'forecast_results.csv',
    index=False
)

# =========================
# PLOT FORECAST
# =========================

fig = model.plot(forecast)

plt.title(
    'AI Revenue Forecast'
)

plt.xlabel(
    'Date'
)

plt.ylabel(
    'Revenue'
)

plt.show()