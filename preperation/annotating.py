import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import argparse
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from datetime import timedelta, datetime

# Constants
CONFIG = {
    "window_length": 600,                  # Must be odd and smaller than the dataset size
    "polyorder": 4,                         # Polynomial order for fitting
    "quantile_increase": 0.94,              # 88th percentile
    "quantile_decrease": 0.12,              # 12th percentile
    "temperature_deviation_holding": 0.5,   # 0.5 °C
    "time_window_holding": 60,              # 1h
    "min_block_size_normalization": 300,    # 10 min
    "minimum_distance_between_peaks": 5400,  # 1.5h
    "number_of_peaks": 5
}

def discover_files(data_dir: str, prefix: str) -> list[str]:
    files = glob.glob(os.path.join(data_dir, f"{prefix}*.csv"))
    return files

def load_and_combine_csv(files: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

def load_and_prepare_data(file_path, single_day):
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Convert 'datetime' column to pandas datetime object
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Filter the data for a specific day
    data = data[data['datetime'].dt.date == pd.to_datetime(single_day).date()]
    return data

def smooth_data(data):
    # Apply Savitzky-Golay filter for smoothing
    data['rolling_mean'] = savgol_filter(data['avg_air_temp'], CONFIG["window_length"], CONFIG["polyorder"])
    return data

def calculate_gradients_and_label(data):
    # Calculate the gradient on the smoothed data
    data['gradient'] = np.gradient(data['rolling_mean'])

    # Define thresholds dynamically based on gradient distribution
    increasing_threshold = data['gradient'].quantile(CONFIG["quantile_increase"])  # 90th percentile
    decreasing_threshold = data['gradient'].quantile(CONFIG["quantile_decrease"])  # 10th percentile

    data['phase'] = 'Nothing'
    data.loc[data['gradient'] > increasing_threshold, 'phase'] = 'Increasing'
    data.loc[data['gradient'] < decreasing_threshold, 'phase'] = 'Decreasing'
    return data

def label_holding_areas(data):
    # Use `find_peaks` with additional parameters to refine peak detection
    peaks_indices, properties = find_peaks(
        data['rolling_mean'],
        prominence=0.1,  # Minimum prominence of peaks
        distance=CONFIG["minimum_distance_between_peaks"],   # Minimum distance between peaks -> 1.5h
        height=None      # Can add a minimum height if required
    )

    # Extract the 5 highest peaks from the identified peaks
    highest_peaks_indices = data.iloc[peaks_indices]['rolling_mean'].nlargest(CONFIG["number_of_peaks"]).index

    # Extend the holding area to include 60 minutes before and after each peak if within -1.5°C of the peak value
    for peak_index in highest_peaks_indices:
        peak_value = data.loc[peak_index, 'rolling_mean']
        peak_time = data.loc[peak_index, 'datetime']

        # Define the time range (60 minutes before and after)
        start_time = peak_time - timedelta(minutes=CONFIG["time_window_holding"])
        end_time = peak_time + timedelta(minutes=CONFIG["time_window_holding"])

        # Mark points as "Holding" if within the time range and within *°C of the peak value
        holding_indices = data.index[
            (data['datetime'] >= start_time) &
            (data['datetime'] <= end_time) &
            (data['rolling_mean'] >= peak_value - CONFIG["temperature_deviation_holding"])
        ]
        data.loc[holding_indices, 'phase'] = 'Holding'
    
    return data, highest_peaks_indices

def normalize_blocks(data):
    # Assign a unique ID to consecutive blocks of the same phase
    data['block'] = (data['phase'] != data['phase'].shift()).cumsum()

    # Calculate the size of each block
    block_sizes = data.groupby('block').size()

    # Normalize by merging small blocks into surrounding larger blocks
    min_block_size = CONFIG["min_block_size_normalization"]
    for block_id, size in block_sizes.items():
        if size < min_block_size:
            indices = data[data['block'] == block_id].index

            if len(indices) > 0:
                previous_index = indices[0] - 1 if indices[0] > 0 else None
                next_index = indices[-1] + 1 if indices[-1] < data.index[-1] else None

                # Assign the phase of the nearest valid neighbor
                if previous_index is not None and previous_index in data.index:
                    data.loc[indices, 'phase'] = data.loc[previous_index, 'phase']
                elif next_index is not None and next_index in data.index:
                    data.loc[indices, 'phase'] = data.loc[next_index, 'phase']
                else:
                    # No valid neighbors; assign 'Nothing'
                    data.loc[indices, 'phase'] = 'Nothing'

    # Drop the temporary block column
    data.drop(columns='block', inplace=True)
    return data


def plot_smoothed_data(data, highest_peaks_indices, day):
    # Plot the rolling mean with normalized heavy slope regions highlighted
    plt.figure(figsize=(15, 6))
    plt.plot(data['datetime'], data['rolling_mean'], color='orange', label=f"Rolling Mean (Window Size = {CONFIG['window_length']})", linewidth=1.5)

    # Highlight normalized heavy slope categories
    for slope_category, color, label in [('Increasing', 'green', 'Increasing'),
                                        ('Decreasing', 'red', 'Decreasing'),
                                        ('Holding', 'purple', 'Holding'),
                                        ('Nothing', 'grey', 'Nothing')]:
        slope_data = data[data['phase'] == slope_category]
        plt.scatter(slope_data['datetime'], slope_data['rolling_mean'], color=color, label=label, s=10, zorder=5)

    # Highlight the top 5 peaks
    plt.scatter(data['datetime'][highest_peaks_indices], data['rolling_mean'][highest_peaks_indices], 
                color='blue', label=f'Top {CONFIG["number_of_peaks"]} Peaks', s=100, marker='x', zorder=5)

    plt.title(f"Rolling Mean with Normalized Heavy Slopes ({day})")
    plt.xlabel("Datetime")
    plt.ylabel("Rolling Mean (°C)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_original_data(data, day):
    # Plot the original data with labels for Increasing, Decreasing, Holding, and Nothing
    plt.figure(figsize=(15, 6))
    plt.plot(data['datetime'], data['avg_air_temp'], color='orange', label="Original Data (avg_air_temp)", linewidth=1.5)

    # Highlight labels on the original data
    for slope_category, color, label in [('Increasing', 'green', 'Increasing'),
                                        ('Decreasing', 'red', 'Decreasing'),
                                        ('Holding', 'purple', 'Holding'),
                                        ('Nothing', 'grey', 'Nothing')]:
        slope_data = data[data['phase'] == slope_category]
        plt.scatter(slope_data['datetime'], slope_data['avg_air_temp'], color=color, label=label, s=10, zorder=5)

    plt.title(f"Original Data (avg_air_temp) with Labels ({day})")
    plt.xlabel("Datetime")
    plt.ylabel("Average Air Temperature (°C)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_entire_week(data):
    # Plot the original data with labels for Increasing, Decreasing, Holding, and Nothing
    plt.figure(figsize=(20, 8))
    plt.plot(data['datetime'], data['avg_air_temp'], color='orange', label="Original Data (avg_air_temp)", linewidth=1.5)

    # Highlight labels on the original data
    for slope_category, color, label in [('Increasing', 'green', 'Increasing'),
                                        ('Decreasing', 'red', 'Decreasing'),
                                        ('Holding', 'purple', 'Holding'),
                                        ('Nothing', 'grey', 'Nothing')]:
        slope_data = data[data['phase'] == slope_category]
        plt.scatter(slope_data['datetime'], slope_data['avg_air_temp'], color=color, label=label, s=10, zorder=5)

    plt.title("Original Data (avg_air_temp) with Labels for the Entire Week")
    plt.xlabel("Datetime")
    plt.ylabel("Average Air Temperature (°C)")
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.show()

def write_week_to_file(file_path, annotated_data):
    annotated_data.to_csv(file_path, index=False)
    print(f"Annotated week data saved to: {file_path}")

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Preprocess CSV files.")
    parser.add_argument("--data_dir", required=True, help="Directory with raw files.")
    args = parser.parse_args()

    data_dir = args.data_dir.lower()
    temp_files = discover_files(data_dir, "temp")
    temp_df = load_and_combine_csv(temp_files)

    # Load the entire dataset to display available dates
    temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])

    # Prompt user to enter start and end dates
    start_date = temp_df['datetime'].min().date()
    end_date = pd.to_datetime("2025-02-06", format="%Y-%m-%d").date()
    #end_date = temp_df['datetime'].max().date()

    current_date = start_date

    annotated_week_data = []

    while current_date <= end_date:
        single_day = current_date.strftime('%Y-%m-%d')
        print(f"Processing data for {single_day}...")

        data = load_and_prepare_data(temp_files[0], single_day)
        data = smooth_data(data)
        
        # Calculate gradients and label data
        data = calculate_gradients_and_label(data)

        # Extend holding area around peaks
        data, highest_peaks_indices = label_holding_areas(data)

        data = normalize_blocks(data)

        # Plot results
        plot_smoothed_data(data, highest_peaks_indices, single_day)
        plot_original_data(data, single_day)

        # Prompt user for config adjustments
        print("Would you like to adjust the configuration for this day? (y/n)")
        if input().lower() == 'y':
            print("Enter new parameters (leave blank to keep current values):")
            for key in CONFIG.keys():
                new_value = input(f"{key} (current: {CONFIG[key]}): ")
                if new_value:
                    CONFIG[key] = type(CONFIG[key])(new_value)

            print("Configuration updated. Re-running for the same day...")
            continue  # Re-run for the same day with updated config

        annotated_week_data.append(data)
        current_date += timedelta(days=1)

    annotated_week_data = pd.concat(annotated_week_data, ignore_index=True)
    write_week_to_file(os.path.join(data_dir, "temp_annotated.csv"), annotated_week_data)
    plot_entire_week(annotated_week_data)

if __name__ == "__main__":
    main()
