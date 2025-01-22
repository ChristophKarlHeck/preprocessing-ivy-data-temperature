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

# Set a 5-minute rolling window
data['rolling_diff'] = data['avg_leaf_temp'].diff().rolling(window=300, min_periods=1).sum()

# Print rolling_diff statistics for debugging
print("Rolling Difference Statistics:")
print(data['rolling_diff'].describe())

# Define thresholds for classification
increase_threshold = 0.4 #0.5  # Adjust this value
decrease_threshold = -0.5 #-0.5  # Adjust this value
hold_threshold = 0.1 # 0.2  # Adjust this value

# Classify phases
data['phase'] = 'Nothing'
data.loc[data['rolling_diff'] > increase_threshold, 'phase'] = 'Increasing'
data.loc[data['rolling_diff'] < decrease_threshold, 'phase'] = 'Decreasing'
data.loc[data['rolling_diff'].abs() <= hold_threshold, 'phase'] = 'Holding'

# Define colors for the phases
phase_colors = {'Nothing': 'gray', 'Increasing': 'red', 'Decreasing': 'blue', 'Holding': 'green'}

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
axes[1].set_title("Classified Phases (5-Min Window Based)")
axes[1].set_xlabel("Datetime")
axes[1].set_ylabel("Temperature (°C)")
axes[1].legend()
axes[1].grid()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
