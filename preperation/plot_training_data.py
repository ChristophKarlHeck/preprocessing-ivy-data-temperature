import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# Use the PGF backend
# matplotlib.use("pgf")

# # Update rcParams
# plt.rcParams.update({
#     "pgf.texsystem": "xelatex",  # Use XeLaTeX
#     "font.family": "sans-serif",  # Use a sans-serif font
#     "font.sans-serif": ["Arial"],  # Specifically use Arial
#     "font.size": 10,  # Set the font size
#     "text.usetex": True,  # Use LaTeX for text rendering
#     "pgf.rcfonts": False,  # Do not override Matplotlib's rc settings
# })

def z_score_normalize(data_slice: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Apply Z-Score normalization to a given data slice.

    Args:
        data_slice (np.ndarray): The input array to normalize.
        factor (float): Scaling factor to adjust the normalized values (default is 1.0).

    Returns:
        np.ndarray: The Z-score normalized array.
    """
    mean = np.mean(data_slice)
    std = np.std(data_slice)

    if std == 0:
        return np.zeros_like(data_slice)  # Avoid division by zero

    return ((data_slice - mean) / std) * factor

def min_max_normalize(data_slice: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Apply Min-Max normalization to a given data slice.

    The normalization transforms the data into the range [0, factor].

    Args:
        data_slice (np.ndarray): The input array to normalize.
        factor (float): Scaling factor to adjust the normalized values (default is 1.0).

    Returns:
        np.ndarray: The min-max normalized array.
    """
    min_val = -200#data_slice.min()
    max_val = 200#data_slice.max()
    range_val = max_val - min_val

    if range_val == 0:
        return np.zeros_like(data_slice)  # Avoid division by zero if all values are the same

    normalized = (data_slice - min_val) / range_val
    return normalized * factor

def adjusted_min_max_normalize(data_slice: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Apply Min-Max normalization to a given data slice.

    The normalization transforms the data into the range [0, factor].

    Args:
        data_slice (np.ndarray): The input array to normalize.
        factor (float): Scaling factor to adjust the normalized values (default is 1.0).

    Returns:
        np.ndarray: The min-max normalized array.
    """
    min_val = data_slice.min()
    max_val = data_slice.max()
    range_val = (max_val/factor) - (min_val/factor)

    if range_val == 0:
        return np.zeros_like(data_slice)  # Avoid division by zero if all values are the same

    normalized = (data_slice - (min_val/factor)) / range_val
    return normalized

def extract_data(data_dir, prefix, before, after, split_minutes):
    # Define file paths
    temp_annotated_path = os.path.join(data_dir, "preprocessed/temp_annotated.csv")
    preprocessed_path = os.path.join(data_dir, f"preprocessed_none_1/{prefix}_preprocessed.csv")
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
    segments_ch1, segments_ch2, all_segments, segments_datetime = [], [], [], []

    nbr_heating_phases = 0
    for start_time in distinct_increasing_starts:
        nbr_heating_phases += 1
        mask = (merged_data['datetime'] >= start_time - time_window_before) & (merged_data['datetime'] <= start_time + time_window_after)
        segment = merged_data.loc[mask, ['datetime', 'CH1_smoothed_scaled', 'CH2_smoothed_scaled']]

        if not segment.empty:
            # Normalize CH1
            ch1_values = segment['CH1_smoothed_scaled'].values
            #ch1_values = min_max_normalize(ch1_values,1000)
            #ch1_values = adjusted_min_max_normalize(ch1_values,1000)
            #ch1_values = z_score_normalize(ch1_values)

            #ch1_values = ch1_values - ch1_values[0]  # Subtract the first value

            # Normalize CH2 (if needed)
            ch2_values = segment['CH2_smoothed_scaled'].values
            #ch2_values = min_max_normalize(ch2_values,1000)
           # ch2_values = adjusted_min_max_normalize(ch2_values,1000)
            #ch2_values = z_score_normalize(ch2_values)
            #ch2_values = ch2_values - ch2_values[0]  # Subtract the first value

            # segments_ch1.append(ch1_values)
            # segments_ch2.append(ch2_values)

            # Snippets
            ###########################################################################################################
            min10 = 600
            n_groups = len(ch1_values) // min10

            result_ch1 = []
            result_ch2 = []
            for i in range(n_groups):
                cut_ch1 = ch1_values[i*min10:(i+1)*min10]
                cut_ch2 = ch2_values[i*min10:(i+1)*min10]

                #print(len(cut_ch1))
                
                res_ch1 = min_max_normalize(cut_ch1, 1)
                res_ch2 = min_max_normalize(cut_ch2, 1)

                result_ch1.extend(res_ch1)
                result_ch2.extend(res_ch2)

            ch1_values = result_ch1
            ch2_values = result_ch2

            ##############################################################################################################
            # Rolling Window
            ###########################################################################################################
            # min10 = 600
            # rolling_distance = min10/100

            # segment_after_ch1 = ch1_values[min10:]
            # segment_after_ch2 = ch2_values[min10:]
            # n_groups = len(ch1_values) // (rolling_distance)

            # result_ch1 = ch1_values[]
            # result_ch2 = []
            # for i in range(n_groups):
            #     cut_ch1 = ch1_values[i*min10:(i+1)*min10]
            #     cut_ch2 = ch2_values[i*min10:(i+1)*min10]
                
            #     res_ch1 = adjusted_min_max_normalize(cut_ch1, 1000)
            #     res_ch2 = adjusted_min_max_normalize(cut_ch2, 1000)

            #     result_ch1.extend(res_ch1)
            #     result_ch2.extend(res_ch2)

            # ch1_values = result_ch1
            # ch2_values = result_ch2

            ##############################################################################################################


            all_segments.append(ch1_values)
            all_segments.append(ch2_values)
            segments_datetime.append(segment['datetime'].values)

    print(nbr_heating_phases)
    # Ensure segments_ch1 and segments_ch2 are NumPy arrays.
    segments_ch1 = np.array(segments_ch1)
    segments_ch2 = np.array(segments_ch2)

    # Calculate the mean (average) segment across all segments.
    # avg_segment_ch1 = np.mean(segments_ch1, axis=0)
    # avg_segment_ch2 = np.mean(segments_ch2, axis=0)

    # avg_segment_ch1 = avg_segment_ch1 - avg_segment_ch1[0]
    # avg_segment_ch2 = avg_segment_ch2 - avg_segment_ch2[0]

    # # Create a time axis for plotting.
    # # 'before' and 'after' should be defined as the time (in minutes) before and after the event.
    # time_axis = np.linspace(-before, after, segments_ch1.shape[1])

    # plt.figure(figsize=(10, 6))

    # # Plot every individual CH1 segment (with some transparency).
    # for segment in segments_ch1:
    #     plt.plot(time_axis, segment, color='red', alpha=0.1)

    # # Optionally, plot the average CH1 segment in a contrasting style.
    # plt.plot(time_axis, avg_segment_ch1, color='black', linewidth=2, label='Average CH1')

    # # Add a vertical line to indicate the event (e.g., start of "Increasing").
    # plt.axvline(0, color='blue', linestyle='--', label='Start of Increasing')

    # plt.xlabel("Time (minutes relative to Increasing start)")
    # plt.ylabel("Scaled Electrical Potential")
    # plt.title("Overlay of All CH1 Trials")
    # plt.legend()
    # plt.show()

    return all_segments #segments_ch1, segments_ch2


if __name__ == "__main__":
    # Load configuration from the JSON file
    with open("config_new.json", "r") as file:
        configs = json.load(file)

    all_segments_ch1 = []
    all_segments_ch2 = []
    all_segments_both = []

    # Iterate over the configurations and call extract_data for each
    for config in configs:
        all_segments = extract_data(
            config["data_dir"],
            config["prefix"],
            config["before"],
            config["after"],
            config["split_minutes"]
        )
        # all_segments_ch1.append(segments_ch1)  # Store as list of measurements
        # all_segments_ch2.append(segments_ch2)
        all_segments_both.append(all_segments)

    # Ensure each segment has the same length and set fixed length to 3600 (600*6)
    max_length = 3600
    time_axis = np.linspace(-30, 30, max_length)
    
    # x_values_ch1 = []
    # y_values_ch1 = []
    # x_values_ch2 = []
    # y_values_ch2 = []

    x_values_both = []
    y_values_both = []

    # for not average indent 2 spaces 
    
    # for segments in all_segments_ch1:
    #     for segment in segments:
    #         if len(segment) > max_length:
    #             segment = segment[:max_length]  # Trim longer segments
    #         time_subset = time_axis[:len(segment)]
    #         x_values_ch1.extend(time_subset)
    #         y_values_ch1.extend(segment)
    
    # for segments in all_segments_ch2:
    #     for segment in segments:
    #         if len(segment) > max_length:
    #             segment = segment[:max_length]  # Trim longer segments
    #         time_subset = time_axis[:len(segment)]
    #         x_values_ch2.extend(time_subset)
    #         y_values_ch2.extend(segment)

    for segments in all_segments_both:
        for segment in segments:
            if len(segment) > max_length:
                segment = segment[:max_length]  # Trim longer segments
            time_subset = time_axis[:len(segment)]
            x_values_both.extend(time_subset)
            y_values_both.extend(segment)
    
    # x_values_ch1 = np.array(x_values_ch1)
    # y_values_ch1 = np.array(y_values_ch1)
    # x_values_ch2 = np.array(x_values_ch2)
    # y_values_ch2 = np.array(y_values_ch2)


    plt.figure(figsize=(3,2))

    # Create a hexbin plot using the combined data
    hb = plt.hexbin(x_values_both, y_values_both, gridsize=50, cmap='Reds', mincnt=1)

    plt.xlabel("Minutes", fontsize=10)
    plt.ylabel("EDP [scaled]", fontsize=10)

    # Add a vertical line at time zero to indicate the start of Heating
    plt.axvline(0, color='black', linestyle='--')

    #plt.legend(loc="lower left", fontsize=8)
    #plt.colorbar(hb)
    plt.tight_layout()
    #plt.show()
    plt.savefig("heatMapMM1.pgf", format="pgf", bbox_inches="tight", pad_inches=0.05)

    
    # # Create subplots for CH1 and CH2
    # fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # # yticks = [-3000, -2000, -1000, 0, 1000, 2000, 3000]
    # # ytick_labels = ["-3000", "-2000", "-1000", "0", "1000", "2000", "3000"]
    
    # axes[0].hexbin(x_values_ch1, y_values_ch1, gridsize=50, cmap='Reds', mincnt=1)
    # axes[0].set_title("Training Data Ch0")
    # axes[0].set_xlabel("Time (minutes relative to Increasing start)")
    # axes[0].set_ylabel("Scaled Electrical Potential (mV)")
    # # axes[0].set_ylim([-3000, 3000])
    # #axes[0].set_yticks(yticks)
    # #axes[0].set_yticklabels(ytick_labels)
    # axes[0].axvline(0, color='blue', linestyle='--', label="Start of Increasing")
    # axes[0].legend()
    
    # axes[1].hexbin(x_values_ch2, y_values_ch2, gridsize=50, cmap='Blues', mincnt=1)
    # axes[1].set_title("Training Data Ch1")
    # axes[1].set_xlabel("Time (minutes relative to Increasing start)")
    # axes[1].set_ylabel("Scaled Electrical Potential (mV)")
    # # axes[1].set_ylim([-3000, 3000])
    # #axes[1].set_yticks(yticks)
    # #axes[1].set_yticklabels(ytick_labels)
    # axes[1].axvline(0, color='blue', linestyle='--', label="Start of Increasing")
    # axes[1].legend()
    
    # plt.colorbar(axes[0].collections[0], ax=axes[0], label="Density of Measurements")
    # plt.colorbar(axes[1].collections[0], ax=axes[1], label="Density of Measurements")
    
    # plt.tight_layout()
    # plt.show()