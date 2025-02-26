import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.colors import LinearSegmentedColormap

def extract_data(data_dir, prefix, before, after, split_minutes):
    # Define file paths
    temp_annotated_path = os.path.join(data_dir, "preprocessed/temp_annotated.csv")
    preprocessed_path = os.path.join(data_dir, f"preprocessed/{prefix}_preprocessed.csv")
    plants_path = os.path.join(data_dir, "plants.csv")
    
    # Load data
    temp_annotated = pd.read_csv(temp_annotated_path, parse_dates=['datetime'])
    if(prefix=="P5" and "8_2025_01_31-2025_02_11" in data_dir):
        temp_annotated = temp_annotated[temp_annotated["datetime"] < "2025-02-07"]
    preprocessed = pd.read_csv(preprocessed_path, parse_dates=['datetime'])
    plants = pd.read_csv(plants_path)
    
    # Identify distinct "Increasing" phase starts (where the previous phase was not "Increasing")
    temp_annotated["prev_phase"] = temp_annotated["phase"].shift(1)
    distinct_increasing_starts = (temp_annotated[
        (temp_annotated["phase"] == "Increasing") & (temp_annotated["prev_phase"] != "Increasing")
    ]['datetime'] + pd.Timedelta(minutes=0)).tolist()
    
    # Merge datasets using nearest datetime to match the preprocessed data with annotations
    merged_data = pd.merge_asof(
        preprocessed.sort_values('datetime'),
        temp_annotated.sort_values('datetime'),
        on='datetime',
        direction='nearest'
    )

    # Extract 60-minute segments around each increasing start
    time_window_before = pd.Timedelta(minutes=before)
    time_window_after = pd.Timedelta(minutes=after)
    segments_ch1, segments_ch2, segments_datetime = [], [], []

    for start_time in distinct_increasing_starts:
        mask = (merged_data['datetime'] >= start_time - time_window_before) & (merged_data['datetime'] <= start_time + time_window_after)
        segment = merged_data.loc[mask, ['datetime', 'CH1_smoothed_scaled', 'CH2_smoothed_scaled']]

        # Normalize CH1
        ch1_values = segment['CH1_smoothed_scaled'].values
        ch1_adjusted = ch1_values - ch1_values[0]  # Subtract the first value

          # Normalize CH2 (if needed)
        ch2_values = segment['CH2_smoothed_scaled'].values
        ch2_adjusted = ch2_values - ch2_values[0]  # Subtract the first value

        segments_ch1.append(ch1_adjusted)
        segments_ch2.append(ch2_adjusted)
        segments_datetime.append(segment['datetime'].values)


    # #Convert to NumPy arrays for processing
    segments_ch1 = np.array(segments_ch1)
    segments_ch2 = np.array(segments_ch2)

    # Create a time axis
    time_axis = np.linspace(-before, after, segments_ch1.shape[1])

    plt.figure(figsize=(10, 6))

    for segment in segments_ch1:
        plt.plot(time_axis, segment, color='red', alpha=0.1)  # Transparent lines

    plt.axvline(0, color='blue', linestyle='--', label='Start of Increasing')
    plt.xlabel("Time (minutes relative to Increasing start)")
    plt.ylabel("Scaled Electrical Potential")
    plt.title("Overlay of All CH1 Trials")
    plt.show()

    x_values = []
    y_values = []

    for segment in segments_ch1:
        time_subset = time_axis[:len(segment)]  # Ensure time axis matches the segment length
        x_values.extend(time_subset)
        y_values.extend(segment)

    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Create hexbin plot
    plt.figure(figsize=(12, 6))
    plt.hexbin(x_values, y_values, gridsize=50, cmap='Reds', mincnt=1)

    # Add colorbar
    plt.colorbar(label="Density of Measurements")

    # Add reference line at time zero
    plt.axvline(0, color='blue', linestyle='--', label="Start of Increasing")

    # Labels and title
    plt.xlabel("Time (minutes relative to Increasing start)")
    plt.ylabel("Scaled Electrical Potential")
    plt.title("Hexbin Density Plot of CH1 Trials")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Load configuration from the JSON file
    with open("config.json", "r") as file:
        configs = json.load(file)

    all_segments_ch1 = []

    # Iterate over the configurations and call extract_data for each
    for config in configs:
        segments = extract_data(
            config["data_dir"],
            config["prefix"],
            config["before"],
            config["after"],
            config["split_minutes"]
        )
        all_segments_ch1.extend(segments)
