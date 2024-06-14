import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the water level data from the CSV file
df = pd.read_csv('water_level_data.csv')

# Convert 'Timestamp' column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Set 'Timestamp' column as the index
df = df.set_index('Timestamp')

# Convert 'Water Level' column to numeric, coerce errors to NaN
df['Water Level'] = pd.to_numeric(df['Water Level'], errors='coerce')

# Drop rows with NaN values
df = df.dropna(subset=['Water Level'])

# Plot the water level data over time
df.plot(y='Water Level', xlabel='Timestamp', ylabel='Water Level', title='Water Level Over Time')
plt.show()

# Perform time series analysis and make predictions using ARIMA modeling
model = ARIMA(df['Water Level'], order=(1, 1, 1))
model_fit = model.fit()

# Forecast future water levels
forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)

# Display the forecasted water levels
print("Forecasted Water Levels:")
print(forecast)

# Check for potential flood risks based on the forecasted water levels
threshold = 450  # Define a threshold for flood risk
for level in forecast:
    if level > threshold:
        print("Potential flood risk detected! Water level exceeds threshold.")
