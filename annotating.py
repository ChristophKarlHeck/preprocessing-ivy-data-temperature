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

# ===== Block Size Approach =====
# block_size = 120
# data['block'] = data.index // block_size

# # Calculate block-level statistics
# block_stats = data.groupby('block')['avg_air_temp'].agg(['mean', 'std', 'min', 'max'])
# block_stats['mean_diff'] = block_stats['mean'].diff().fillna(0)  # Fill NaN for the first block
# block_stats['phase_block'] = 'NotYet'
# block_stats.loc[block_stats['mean_diff'].abs() < 0.17, 'phase_block'] = "Nothing"
# block_stats.loc[block_stats['mean_diff'] >= 0.17, 'phase_block'] = 'Increasing'
# block_stats.loc[block_stats['mean_diff'] <= -0.17, 'phase_block'] = 'Decreasing'
# block_stats.loc[(block_stats['mean_diff'].abs() < 0.17) & (block_stats['mean'] > 24), 'phase_block'] = 'Holding'

# # Map block statistics to the original data
# block_stats = block_stats.reset_index()
# data['phase_block'] = data['block'].map(block_stats.set_index('block')['phase_block'])

# # Plot
# block_timestamps = data.groupby('block')['datetime'].first().reset_index(drop=True)
# plt.figure(figsize=(15, 6))
# plt.plot(block_timestamps, block_stats['mean_diff'], color='orange', label="Mean Difference", linewidth=1.5)
# plt.title("Mean Difference Between Blocks")
# plt.xlabel("Datetime")
# plt.ylabel("Mean Difference (°C)")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(15, 6))
# plt.plot(block_timestamps, block_stats['mean_diff'], color='orange', label="Mean Difference", linewidth=1.5)

# # Identify 5 highest and 5 lowest peaks
# highest_peaks = block_stats['mean_diff'].nlargest(5)
# lowest_peaks = block_stats['mean_diff'].nsmallest(5)

# # Plot highest and lowest peaks
# plt.scatter(block_timestamps[highest_peaks.index], highest_peaks, color='blue', label='Top 5 Highest Peaks', s=100, zorder=5)
# plt.scatter(block_timestamps[lowest_peaks.index], lowest_peaks, color='red', label='Top 5 Lowest Peaks', s=100, zorder=5)

# plt.title("Mean Difference Between Blocks with Top 5 Highest and Lowest Peaks")
# plt.xlabel("Datetime")
# plt.ylabel("Mean Difference (°C)")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

# ===== Window Sliding Approach ===== Focusing on decrease
window_size = 1200
data['rolling_mean'] = data['avg_air_temp'].rolling(window=window_size, min_periods=1).mean()
data['previous_mean'] = data['rolling_mean'].shift()
data['trend_window'] = data['rolling_mean'] - data['previous_mean']

data['phase_window'] = 'NotYet'
data.loc[data['trend_window'].abs() < 0.0008, 'phase_window'] = "Nothing"
data.loc[data['trend_window'] >= 0.0008, 'phase_window'] = 'Increasing'
data.loc[data['trend_window'] <= -0.0008, 'phase_window'] = 'Decreasing'
data.loc[(data['trend_window'].abs() < 0.0008) & (data['rolling_mean'] > 24), 'phase_window'] = 'Holding'

# Use `find_peaks` with additional parameters to refine peak detection
peaks_indices, properties = find_peaks(
    data['rolling_mean'],
    prominence=0.1,  # Minimum prominence of peaks
    distance=5400,   # Minimum distance between peaks
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
increasing_threshold = data['gradient'].quantile(0.90)  # 90th percentile
decreasing_threshold = data['gradient'].quantile(0.10)  # 10th percentile

# Mark regions of heavy increasing and decreasing slopes
data['heavy_slope'] = 'None'
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

# Plot the rolling mean with both peaks and heavy slope regions highlighted
plt.figure(figsize=(15, 6))
plt.plot(data['datetime'], data['rolling_mean'], color='orange', label="Rolling Mean (Window Size = 1200)", linewidth=1.5)

# Highlight heavy increasing slopes
heavy_increasing = data[data['heavy_slope'] == 'Heavy Increasing']
plt.scatter(heavy_increasing['datetime'], heavy_increasing['rolling_mean'], 
            color='green', label='Heavy Increasing', s=10, zorder=5)

# Highlight heavy decreasing slopes
heavy_decreasing = data[data['heavy_slope'] == 'Heavy Decreasing']
plt.scatter(heavy_decreasing['datetime'], heavy_decreasing['rolling_mean'], 
            color='red', label='Heavy Decreasing', s=10, zorder=5)

# Highlight heavy increasing slopes
heavy_increasing = data[data['heavy_slope'] == 'Holding']
plt.scatter(heavy_increasing['datetime'], heavy_increasing['rolling_mean'], 
            color='purple', label='Holding', s=10, zorder=5)

# Highlight the top 5 peaks
plt.scatter(data['datetime'][highest_peaks_indices], data['rolling_mean'][highest_peaks_indices], 
            color='blue', label='Top 5 Peaks', s=100, marker='x', zorder=5)

plt.title("Rolling Mean with Top 5 Peaks and Heavy Slopes")
plt.xlabel("Datetime")
plt.ylabel("Rolling Mean (°C)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Combine the peak and heavy slope data for display
# combined_slope_and_peaks_data = pd.concat([
#     heavy_increasing[['datetime', 'rolling_mean', 'gradient', 'heavy_slope']],
#     heavy_decreasing[['datetime', 'rolling_mean', 'gradient', 'heavy_slope']],
#     highest_peaks_data[['datetime', 'rolling_mean']].rename(columns={'rolling_mean': 'peak_value'}),
# ])

# tools.display_dataframe_to_user(name="Combined Peaks and Heavy Slope Data", dataframe=combined_slope_and_peaks_data)

# ===== Derivative-Based Approach =====
# data['time_diff'] = data['datetime'].diff().dt.total_seconds()  # Time difference in seconds
# data['derivative'] = data['avg_leaf_temp'].diff() / data['time_diff']
# data['phase_derivative'] = 'NotYet'
# data.loc[data['derivative'] > 0.05, 'phase_derivative'] = 'Increasing'
# data.loc[data['derivative'] < -0.05, 'phase_derivative'] = 'Decreasing'
# data.loc[data['derivative'].abs() <= 0.05, 'phase_derivative'] = 'Holding'

# # Plot
# plt.figure(figsize=(15, 6))
# plt.plot(data['datetime'], data['derivative'], color='orange', label="Trend Window", linewidth=1.5)
# plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Reference line for zero trend
# plt.title("Trend Window (Derivative-Based Approach)")
# plt.xlabel("Datetime")
# plt.ylabel("Trend (°C per Window)")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()


# ===== Combined Phases =====
# Priority: Holding > Increasing > Decreasing > Nothing
# phase_priority = {'Holding': 1, 'Increasing': 1, 'Decreasing': 1, 'Nothing': 1}
# data['combined_phase'] = data[['phase_block', 'phase_window', 'phase_derivative']].apply(
#     lambda row: max([x for x in row if x in phase_priority], key=lambda x: phase_priority[x]), axis=1
# )

# ===== Visualization =====
# # Define colors for phases
# phase_colors = {'Nothing': 'gray', 'Increasing': 'red', 'Decreasing': 'blue', 'Holding': 'green', 'NotYet': 'yellow'}

# # Create subplots for each method
# fig, axes = plt.subplots(4, 1, figsize=(15, 24), sharex=True)

# # Plot block-level phases
# for phase, color in phase_colors.items():
#     phase_data = data[data['phase_block'] == phase]
#     axes[0].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
# axes[0].set_title("Block-Based Phases")
# axes[0].set_ylabel("Temperature (°C)")
# axes[0].legend()
# axes[0].grid()

# # Plot rolling window phases
# for phase, color in phase_colors.items():
#     phase_data = data[data['phase_window'] == phase]
#     axes[1].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
# axes[1].set_title("Window Sliding Phases")
# axes[1].set_ylabel("Temperature (°C)")
# axes[1].legend()
# axes[1].grid()

# # Plot derivative-based phases
# for phase, color in phase_colors.items():
#     phase_data = data[data['phase_derivative'] == phase]
#     axes[2].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
# axes[2].set_title("Derivative-Based Phases")
# axes[2].set_ylabel("Temperature (°C)")
# axes[2].legend()
# axes[2].grid()

# # Plot combined phases
# for phase, color in phase_colors.items():
#     phase_data = data[data['combined_phase'] == phase]
#     axes[3].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
# axes[3].set_title("Combined Phases")
# axes[3].set_xlabel("Datetime")
# axes[3].set_ylabel("Temperature (°C)")
# axes[3].legend()
# axes[3].grid()

# # Adjust layout and show the plot
# plt.tight_layout()
# plt.show()