import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import find_peaks
import numpy as np
from datetime import timedelta

# Load the CSV file
file_path = "/home/chris/experiment_data/5_09.01.25-15.01.25/preprocessed/temperature_preprocessed.csv"
data = pd.read_csv(file_path)

# Convert 'datetime' column to pandas datetime object
data['datetime'] = pd.to_datetime(data['datetime'])

# Filter the data for a specific day
single_day = '2025-01-13'
data = data[data['datetime'].dt.date == pd.to_datetime(single_day).date()]

# ===== Window Sliding Approach ===== Focusing on decrease
window_size = 1200
data['rolling_mean'] = data['avg_air_temp'].rolling(window=window_size, min_periods=1).mean()

# Use `find_peaks` with additional parameters to refine peak detection
peaks_indices, properties = find_peaks(
    data['rolling_mean'],
    prominence=0.1,  # Minimum prominence of peaks
    distance=5400,   # Minimum distance between peaks -> 1.5h
    height=None      # Can add a minimum height if required
)

# Extract the 5 highest peaks from the identified peaks
highest_peaks_indices = data.iloc[peaks_indices]['rolling_mean'].nlargest(5).index

print(highest_peaks_indices)

# Extract data for the 5 highest peaks
highest_peaks_data = data.loc[highest_peaks_indices, ['datetime', 'avg_air_temp', 'rolling_mean']]

print(highest_peaks_data)

# Recalculate the gradient and mark heavy slopes to ensure the column exists
data['gradient'] = np.gradient(data['rolling_mean'])

# Define thresholds dynamically based on gradient distribution
increasing_threshold = data['gradient'].quantile(0.88)  # 90th percentile
decreasing_threshold = data['gradient'].quantile(0.12)  # 10th percentile

# Mark regions of heavy increasing and decreasing slopes
data['heavy_slope'] = 'Nothing'
data.loc[data['gradient'] > increasing_threshold, 'heavy_slope'] = 'Heavy Increasing'
data.loc[data['gradient'] < decreasing_threshold, 'heavy_slope'] = 'Heavy Decreasing'

# Extend the holding area to include 60 minutes before and after each peak if within -1.5°C of the peak value
for peak_index in highest_peaks_indices:
    peak_value = data.loc[peak_index, 'rolling_mean']
    peak_time = data.loc[peak_index, 'datetime']

    # Define the time range (60 minutes before and after)
    start_time = peak_time - timedelta(minutes=60)
    end_time = peak_time + timedelta(minutes=60)

    # Mark points as "Holding" if within the time range and within -1.5°C of the peak value
    holding_indices = data.index[
        (data['datetime'] >= start_time) &
        (data['datetime'] <= end_time) &
        (data['rolling_mean'] >= peak_value - 1)
    ]
    data.loc[holding_indices, 'heavy_slope'] = 'Holding'

# ===== Normalization =====
# Assign a unique ID to consecutive blocks of the same slope
data['block'] = (data['heavy_slope'] != data['heavy_slope'].shift()).cumsum()

# Calculate the size of each block
block_sizes = data.groupby('block').size()

# Normalize by merging small blocks into surrounding larger blocks
min_block_size = 300  # Define a minimum block size for normalization (5 min)
for block_id, size in block_sizes.items():
    if size < min_block_size:
        # Find indices of the current block
        indices = data[data['block'] == block_id].index
        # Assign the label of the nearest neighboring block
        if indices[0] > 0:
            data.loc[indices, 'heavy_slope'] = data.loc[indices[0] - 1, 'heavy_slope']
        elif indices[-1] < len(data) - 1:
            data.loc[indices, 'heavy_slope'] = data.loc[indices[-1] + 1, 'heavy_slope']

# Drop the temporary block column
data.drop(columns='block', inplace=True)

# Plot the rolling mean with normalized heavy slope regions highlighted
plt.figure(figsize=(15, 6))
plt.plot(data['datetime'], data['rolling_mean'], color='orange', label="Rolling Mean (Window Size = 1200)", linewidth=1.5)

# Highlight normalized heavy slope categories
for slope_category, color, label in [('Heavy Increasing', 'green', 'Heavy Increasing'),
                                     ('Heavy Decreasing', 'red', 'Heavy Decreasing'),
                                     ('Holding', 'purple', 'Holding'),
                                     ('Nothing', 'grey', 'Nothing')]:
    slope_data = data[data['heavy_slope'] == slope_category]
    plt.scatter(slope_data['datetime'], slope_data['rolling_mean'], color=color, label=label, s=10, zorder=5)

# Highlight the top 5 peaks
plt.scatter(data['datetime'][highest_peaks_indices], data['rolling_mean'][highest_peaks_indices], 
            color='blue', label='Top 5 Peaks', s=100, marker='x', zorder=5)

plt.title("Rolling Mean with Normalized Heavy Slopes")
plt.xlabel("Datetime")
plt.ylabel("Rolling Mean (°C)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot the original data with labels for Increasing, Decreasing, Holding, and Nothing
plt.figure(figsize=(15, 6))
plt.plot(data['datetime'], data['avg_air_temp'], color='orange', label="Original Data (avg_air_temp)", linewidth=1.5)

# Highlight labels on the original data
for slope_category, color, label in [('Heavy Increasing', 'green', 'Heavy Increasing'),
                                     ('Heavy Decreasing', 'red', 'Heavy Decreasing'),
                                     ('Holding', 'purple', 'Holding'),
                                     ('Nothing', 'grey', 'Nothing')]:
    slope_data = data[data['heavy_slope'] == slope_category]
    plt.scatter(slope_data['datetime'], slope_data['avg_air_temp'], color=color, label=label, s=10, zorder=5)

plt.title("Original Data (avg_air_temp) with Labels")
plt.xlabel("Datetime")
plt.ylabel("Average Air Temperature (°C)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
