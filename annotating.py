import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "/home/chris/experiment_data/5_09.01.25-15.01.25/preprocessed/temperature_preprocessed.csv"
data = pd.read_csv(file_path)

# Convert 'datetime' column to pandas datetime object
data['datetime'] = pd.to_datetime(data['datetime'])

# Filter the data for a specific day
single_day = '2025-01-13'
data = data[data['datetime'].dt.date == pd.to_datetime(single_day).date()]

# Set a sliding window size
window_size = 300  # Number of points in the sliding window (adjust as needed)

# Calculate the rolling mean for trend detection
data['rolling_mean'] = data['avg_air_temp'].rolling(window=window_size, min_periods=1).mean()
data['rolling_std'] = data['avg_air_temp'].rolling(window=window_size, min_periods=1).std()
data['rolling_min'] = data['avg_air_temp'].rolling(window=window_size, min_periods=1).min()
data['rolling_max'] = data['avg_air_temp'].rolling(window=window_size, min_periods=1).max()

# Create subplots for each rolling statistic
fig, axes = plt.subplots(4, 1, figsize=(15, 18), sharex=True)

# Plot rolling mean
axes[0].plot(data['datetime'], data['rolling_mean'], color='orange', label="Rolling Mean", linewidth=1.5)
axes[0].set_title("Rolling Mean")
axes[0].set_ylabel("Temperature (°C)")
axes[0].legend()
axes[0].grid()

# Plot rolling standard deviation
axes[1].plot(data['datetime'], data['rolling_std'], color='blue', label="Rolling Std Dev", linewidth=1.5)
axes[1].set_title("Rolling Standard Deviation")
axes[1].set_ylabel("Std Dev (°C)")
axes[1].legend()
axes[1].grid()

# Plot rolling minimum
axes[2].plot(data['datetime'], data['rolling_min'], color='green', label="Rolling Min", linewidth=1.5)
axes[2].set_title("Rolling Minimum")
axes[2].set_ylabel("Temperature (°C)")
axes[2].legend()
axes[2].grid()

# Plot rolling maximum
axes[3].plot(data['datetime'], data['rolling_max'], color='red', label="Rolling Max", linewidth=1.5)
axes[3].set_title("Rolling Maximum")
axes[3].set_xlabel("Datetime")
axes[3].set_ylabel("Temperature (°C)")
axes[3].legend()
axes[3].grid()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


# Shift the rolling mean to compare with the previous window
data['previous_mean'] = data['rolling_mean'].shift()

# Calculate the actual trend (difference between current and previous rolling means)
data['trend'] = data['rolling_mean'] - data['previous_mean']


plt.figure(figsize=(15, 6))
plt.plot(data['datetime'], data['trend'], color='orange', label="Rolling Mean (Window=300)", linewidth=1.5)
plt.title("Rolling Mean of Average Air Temperature")
plt.xlabel("Datetime")
plt.ylabel("Rolling Mean (°C)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Classify phases based on trend
data['phase'] = 'NotYet'
data.loc[data['trend'].abs() < 0.0013, 'phase'] = "Nothing"
data.loc[data['trend'] >= 0.0013, 'phase'] = 'Increasing'
data.loc[data['trend'] <= -0.013, 'phase'] = 'Decreasing'
data.loc[(data['trend'].abs() < 0.0013) & (data['rolling_mean'] > 24), 'phase'] = 'Holding'

# Define colors for the phases
phase_colors = {'Nothing': 'gray', 'Increasing': 'red', 'Decreasing': 'blue', 'Holding': 'green', "NotYet":'yellow'}

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# Subplot 1: Original temperature data
axes[0].plot(data['datetime'], data['avg_leaf_temp'], color='black', label="Avg Leaf Temp", linewidth=1.5)
axes[0].set_title("Original Temperature Data")
axes[0].set_ylabel("Temperature (°C)")
axes[0].legend()
axes[0].grid()

# Subplot 2: Classified phases
for phase, color in phase_colors.items():
    phase_data = data[data['phase'] == phase]
    axes[1].scatter(phase_data['datetime'], phase_data['avg_leaf_temp'], color=color, label=phase, s=10)
axes[1].set_title("Classified Phases Based on Trend")
axes[1].set_xlabel("Datetime")
axes[1].set_ylabel("Temperature (°C)")
axes[1].legend()
axes[1].grid()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
