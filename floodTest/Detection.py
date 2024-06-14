import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the water level data from the CSV file
df = pd.read_csv('water_level_data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index('Timestamp')

# Explore the data and plot the water level over time
df.plot()
plt.title('Water Level Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Water Level')
plt.show()

# Perform time series analysis and make predictions using ARIMA modeling
model = ARIMA(df['Water Level'], order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)

# Display the forecasted water levels
print("Forecasted Water Levels:")
print(forecast)

# Check for potential flood risks based on the forecasted water levels
threshold = 450  # Define a threshold for flood risk
for level in forecast:
    if level > threshold:
        print("Potential flood risk detected! Water level exceeds threshold.")

# Additional analysis and actions can be added based on the forecasted data