import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load the CSV file
file_path = "/home/chris/experiment_data/5_09.01.25-15.01.25/preprocessed/temperature_preprocessed.csv"
data = pd.read_csv(file_path)

# Convert 'datetime' column to pandas datetime object
data['datetime'] = pd.to_datetime(data['datetime'])

# Filter the data for a specific day
single_day = '2025-01-13'
data = data[data['datetime'].dt.date == pd.to_datetime(single_day).date()]

# ===== Block Size Approach =====
block_size = 120
data['block'] = data.index // block_size

# Calculate block-level statistics
block_stats = data.groupby('block')['avg_air_temp'].agg(['mean', 'std', 'min', 'max'])
block_stats['mean_diff'] = block_stats['mean'].diff().fillna(0)  # Fill NaN for the first block
block_stats['phase_block'] = 'NotYet'
block_stats.loc[block_stats['mean_diff'].abs() < 0.17, 'phase_block'] = "Nothing"
block_stats.loc[block_stats['mean_diff'] >= 0.17, 'phase_block'] = 'Increasing'
block_stats.loc[block_stats['mean_diff'] <= -0.17, 'phase_block'] = 'Decreasing'
block_stats.loc[(block_stats['mean_diff'].abs() < 0.17) & (block_stats['mean'] > 24), 'phase_block'] = 'Holding'

# Map block statistics to the original data
block_stats = block_stats.reset_index()
data['phase_block'] = data['block'].map(block_stats.set_index('block')['phase_block'])

# ===== Window Sliding Approach =====
window_size = 300
data['rolling_mean'] = data['avg_air_temp'].rolling(window=window_size, min_periods=1).mean()
data['rolling_std'] = data['avg_air_temp'].rolling(window=window_size, min_periods=1).std()
data['trend_window'] = data['rolling_mean'].diff().fillna(0)
data['phase_window'] = 'NotYet'
data.loc[data['trend_window'].abs() < 0.0013, 'phase_window'] = "Nothing"
data.loc[data['trend_window'] >= 0.0013, 'phase_window'] = 'Increasing'
data.loc[data['trend_window'] <= -0.013, 'phase_window'] = 'Decreasing'
data.loc[(data['trend_window'].abs() < 0.0013) & (data['rolling_mean'] > 24), 'phase_window'] = 'Holding'

# ===== Derivative-Based Approach =====
data['rolling_diff'] = data['avg_leaf_temp'].diff().rolling(window=300, min_periods=1).sum().fillna(0)
data['phase_derivative'] = 'NotYet'
data.loc[data['rolling_diff'] > 0.4, 'phase_derivative'] = 'Increasing'
data.loc[data['rolling_diff'] < -0.4, 'phase_derivative'] = 'Decreasing'
data.loc[data['rolling_diff'].abs() <= 0.4, 'phase_derivative'] = 'Holding'

# Check for NaN values in the phase-related columns
nan_rows = data[data[['phase_block', 'phase_window', 'phase_derivative']].isna().any(axis=1)]
if not nan_rows.empty:
    print("Warning: Found NaN values in the phase columns!")
    print(nan_rows)

# ===== Combined Phases =====
# Priority: Holding > Increasing > Decreasing > Nothing
phase_priority = {'Holding': 1, 'Increasing': 1, 'Decreasing': 1, 'Nothing': 1}
data['combined_phase'] = data[['phase_block', 'phase_window', 'phase_derivative']].apply(
    lambda row: max([x for x in row if x in phase_priority], key=lambda x: phase_priority[x]), axis=1
)

# ===== Visualization =====
# Define colors for phases
phase_colors = {'Nothing': 'gray', 'Increasing': 'red', 'Decreasing': 'blue', 'Holding': 'green'}

# Create subplots for each method
fig, axes = plt.subplots(4, 1, figsize=(15, 24), sharex=True)

# Plot block-level phases
for phase, color in phase_colors.items():
    phase_data = data[data['phase_block'] == phase]
    axes[0].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
axes[0].set_title("Block-Based Phases")
axes[0].set_ylabel("Temperature (°C)")
axes[0].legend()
axes[0].grid()

# Plot rolling window phases
for phase, color in phase_colors.items():
    phase_data = data[data['phase_window'] == phase]
    axes[1].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
axes[1].set_title("Window Sliding Phases")
axes[1].set_ylabel("Temperature (°C)")
axes[1].legend()
axes[1].grid()

# Plot derivative-based phases
for phase, color in phase_colors.items():
    phase_data = data[data['phase_derivative'] == phase]
    axes[2].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
axes[2].set_title("Derivative-Based Phases")
axes[2].set_ylabel("Temperature (°C)")
axes[2].legend()
axes[2].grid()

# Plot combined phases
for phase, color in phase_colors.items():
    phase_data = data[data['combined_phase'] == phase]
    axes[3].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
axes[3].set_title("Combined Phases")
axes[3].set_xlabel("Datetime")
axes[3].set_ylabel("Temperature (°C)")
axes[3].legend()
axes[3].grid()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
