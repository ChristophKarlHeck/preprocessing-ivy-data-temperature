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

        if not segment.empty:
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
    # segments_ch1 = np.array(segments_ch1)
    # segments_ch2 = np.array(segments_ch2)

    # Create a time axis
    # time_axis = np.linspace(-before, after, segments_ch1.shape[1])

    # plt.figure(figsize=(10, 6))

    # for segment in segments_ch1:
    #     plt.plot(time_axis, segment, color='red', alpha=0.1)  # Transparent lines

    # plt.axvline(0, color='blue', linestyle='--', label='Start of Increasing')
    # plt.xlabel("Time (minutes relative to Increasing start)")
    # plt.ylabel("Scaled Electrical Potential")
    # plt.title("Overlay of All CH1 Trials")
    # plt.show()

    return segments_ch1, segments_ch2


if __name__ == "__main__":
    # Load configuration from the JSON file
    with open("config.json", "r") as file:
        configs = json.load(file)

    all_segments_ch1 = []
    all_segments_ch2 = []

    # Iterate over the configurations and call extract_data for each
    for config in configs:
        segments_ch1, segments_ch2 = extract_data(
            config["data_dir"],
            config["prefix"],
            config["before"],
            config["after"],
            config["split_minutes"]
        )
        all_segments_ch1.append(segments_ch1)  # Store as list of measurements
        all_segments_ch2.append(segments_ch2)

    # Ensure each segment has the same length and set fixed length to 3600 (600*6)
    max_length = 3600
    time_axis = np.linspace(-30, 30, max_length)
    
    x_values_ch1 = []
    y_values_ch1 = []
    x_values_ch2 = []
    y_values_ch2 = []
    
    for segments in all_segments_ch1:
        for segment in segments:
            if len(segment) > max_length:
                segment = segment[:max_length]  # Trim longer segments
            time_subset = time_axis[:len(segment)]
            x_values_ch1.extend(time_subset)
            y_values_ch1.extend(segment)
    
    for segments in all_segments_ch2:
        for segment in segments:
            if len(segment) > max_length:
                segment = segment[:max_length]  # Trim longer segments
            time_subset = time_axis[:len(segment)]
            x_values_ch2.extend(time_subset)
            y_values_ch2.extend(segment)
    
    x_values_ch1 = np.array(x_values_ch1)
    y_values_ch1 = np.array(y_values_ch1)
    x_values_ch2 = np.array(x_values_ch2)
    y_values_ch2 = np.array(y_values_ch2)
    
    # Create subplots for CH1 and CH2
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    yticks = [-300, -200, -100, 0, 100, 200, 300]
    ytick_labels = ["-300", "-200", "-100", "0", "100", "200", "300"]
    
    axes[0].hexbin(x_values_ch1, y_values_ch1, gridsize=100, cmap='Reds', mincnt=1)
    axes[0].set_title("Training Data Ch0")
    axes[0].set_xlabel("Time (minutes relative to Increasing start)")
    axes[0].set_ylabel("Scaled Electrical Potential (mV)")
    axes[0].set_ylim([-300, 300])
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(ytick_labels)
    axes[0].axvline(0, color='blue', linestyle='--', label="Start of Increasing")
    axes[0].legend()
    
    axes[1].hexbin(x_values_ch2, y_values_ch2, gridsize=100, cmap='Blues', mincnt=1)
    axes[1].set_title("Training Data Ch1")
    axes[1].set_xlabel("Time (minutes relative to Increasing start)")
    axes[1].set_ylabel("Scaled Electrical Potential (mV)")
    axes[1].set_ylim([-300, 300])
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels(ytick_labels)
    axes[1].axvline(0, color='blue', linestyle='--', label="Start of Increasing")
    axes[1].legend()
    
    plt.colorbar(axes[0].collections[0], ax=axes[0], label="Density of Measurements")
    plt.colorbar(axes[1].collections[0], ax=axes[1], label="Density of Measurements")
    
    plt.tight_layout()
    plt.show()