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

    # Create subplots for CH1, CH2, and temperature with phases
    # fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # # Plot CH1
    # axes[0].plot(merged_data['datetime'], merged_data['CH1_smoothed_scaled'], color='black', label='CH1')
    # axes[0].set_title('CH1 Signal')
    # axes[0].set_ylabel('Preprocessed Data')
    # axes[0].grid()
    # axes[0].legend()
    
    # # Plot CH2
    # axes[1].plot(merged_data['datetime'], merged_data['CH2_smoothed_scaled'], color='black', label='CH2')
    # axes[1].set_title('CH2 Signal')
    # axes[1].set_ylabel('Preprocessed Data')
    # axes[1].grid()
    # axes[1].legend()
    
    # # Plot Temperature with Phase Colors
    # for phase, color in [('Increasing', 'green'), ('Decreasing', 'red'), ('Holding', 'purple'), ('Nothing', 'grey')]:
    #     phase_data = merged_data[merged_data['phase'] == phase]
    #     axes[2].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
    
    # axes[2].set_title('Temperature with Phases')
    # axes[2].set_ylabel('Temperature (°C)')
    # axes[2].grid()
    # axes[2].legend(loc='upper right')
    
    # # Plot vertical lines and highlight domain for distinct increasing phase starts
    # for i, ax in enumerate(axes):
    #     for j, start_time in enumerate(distinct_increasing_starts):
    #         label = "Start of Increasing" if j == 0 else None
    #         ax.axvline(start_time, color='blue', linestyle='--', linewidth=1.5, label=label)
            
    #         # Highlighting domain (before and after)
    #         ax.axvspan(start_time - pd.Timedelta(minutes=before), start_time + pd.Timedelta(minutes=after), 
    #                    color='blue', alpha=0.2)
    
    # # Add legend only once per plot
    # axes[0].legend(loc='upper right')
    # axes[1].legend(loc='upper right')
    # axes[2].legend(loc='upper right')
    
    # plt.xlabel('Datetime')
    # plt.tight_layout()
    
    # # Display the plot
    # plt.show()

    # Extract 60-minute segments around each increasing start
    time_window_before = pd.Timedelta(minutes=before)
    time_window_after = pd.Timedelta(minutes=after)
    segments_ch1, segments_ch2, segments_datetime = [], [], []

    for start_time in distinct_increasing_starts:
        mask = (merged_data['datetime'] >= start_time - time_window_before) & (merged_data['datetime'] <= start_time + time_window_after)
        segment = merged_data.loc[mask, ['datetime', 'CH1_smoothed_scaled', 'CH2_smoothed_scaled']]
        segments_ch1.append(segment['CH1_smoothed_scaled'].values)
        segments_ch2.append(segment['CH2_smoothed_scaled'].values)
        segments_datetime.append(segment['datetime'].values)

    print(len(segments_ch1))
    results = []
    num_slices = int((before+after)/split_minutes)  # Each segment is split into 6 slices (10 min each)
    samples_per_slice = int(((before+after)*60)/num_slices) # 10 minutes at 1Hz sampling

    for segment_index, (segment_ch1, segment_ch2, segment_datetime) in enumerate(zip(segments_ch1, segments_ch2, segments_datetime)):
        plant_name = plants.iloc[0][f'{prefix}']  # Extract plant name
        
        for slice_index in range(num_slices):
            start_idx = slice_index * samples_per_slice
            end_idx = start_idx + samples_per_slice
            
            # Extract the 10-min slice for both CH1 and CH2
            

            # Determine the Heat label
            heat = 0 if slice_index < int((before/split_minutes)) else 1  # First 3 slices -> 0, Last 3 slices -> 1

            # Construct row with 600 data points for both channels + metadata
            slice_data_ch1 = segment_ch1[start_idx:end_idx]
            row_ch0 = {
                'Start_Datetime': segment_datetime[start_idx],
                'End_Datetime': segment_datetime[end_idx],
                'Plant': plant_name,
                'Channel': 0,
                'Heat': heat
            }
            row_ch0.update({f'val_{i}': slice_data_ch1[i] for i in range(samples_per_slice)})  # CH1 values
            results.append(row_ch0)

            slice_data_ch2 = segment_ch2[start_idx:end_idx]
            row_ch1 = {
                'Start_Datetime': segment_datetime[start_idx],
                'End_Datetime': segment_datetime[end_idx],
                'Plant': plant_name,
                'Channel': 1,
                'Heat': heat
            }
            row_ch1.update({f'val_{i}': slice_data_ch2[i] for i in range(samples_per_slice)})  # CH2 values

            results.append(row_ch1)

    print(len(results))

    # Convert to DataFrame and save to CSV
    df_results = pd.DataFrame(results)
    output_dir = os.path.join(data_dir, "training_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prefix}_ready_to_train.csv")
    df_results.to_csv(output_path, index=False)

    print(f"Processed slices stored in {output_path}")

    #Convert to NumPy arrays for processing
    segments_ch1 = np.array(segments_ch1)
    segments_ch2 = np.array(segments_ch2)

    mean_signal = np.mean(segments_ch1, axis=0)
    std_signal = np.std(segments_ch1, axis=0)

    # Create a time axis
    time_axis = np.linspace(-before, after, segments_ch1.shape[1])

    # Plot mean signal with shaded standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, mean_signal, color='black', label='Mean Signal')
    plt.fill_between(time_axis, mean_signal - std_signal, mean_signal + std_signal, color='red', alpha=0.2)

    plt.axvline(0, color='blue', linestyle='--', label='Start of Increasing')
    plt.xlabel("Time (minutes relative to Increasing start)")
    plt.ylabel("Scaled Electrical Potential")
    plt.title("Overlayed Mean CH1 Signal with Variability")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))

    for segment in segments_ch1:
        plt.plot(time_axis, segment, color='red', alpha=0.1)  # Transparent lines

    plt.axvline(0, color='blue', linestyle='--', label='Start of Increasing')
    plt.xlabel("Time (minutes relative to Increasing start)")
    plt.ylabel("Scaled Electrical Potential")
    plt.title("Overlay of All CH1 Trials")
    plt.show()

    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.kdeplot(segments_ch1.flatten(), bw_adjust=0.5, fill=True, color="red")

    plt.xlabel("Signal Intensity")
    plt.ylabel("Density")
    plt.title("Density Distribution of CH1 Segments")
    plt.show()

    from scipy.ndimage import gaussian_filter

    #Compute the mean response across all segments (axis=0 means average over trials)
    mean_signal = np.mean(segments_ch1, axis=0)

    # Compute the standard deviation to show variability
    std_signal = np.std(segments_ch1, axis=0)

    # Apply absolute values for intensity scaling
    intensity_data = np.abs(segments_ch1)

    # Sort segments by max intensity for better structure
    sorted_indices = np.argsort(-np.max(intensity_data, axis=1))
    sorted_data = intensity_data[sorted_indices, :]

    # Smooth the data to reveal trends without excessive noise
    smoothed_data = gaussian_filter(sorted_data, sigma=(1, 1))

    # Create time axis
    time_axis = np.linspace(-before, after, segments_ch1.shape[1])

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    im = plt.imshow(smoothed_data, aspect='auto', extent=[time_axis[0], time_axis[-1], 0, segments_ch1.shape[0]], 
                    origin='lower', cmap='magma', vmin=0, vmax=np.percentile(intensity_data, 95)) 

    plt.colorbar(label='Absolute Signal Intensity')
    plt.xlabel("Time (minutes relative to Increasing start)")
    plt.ylabel("Segment Index (Sorted by Max Intensity)")
    plt.title("Optimized CH1 Heatmap with Enhanced Pattern Visibility")

    # Overlay mean signal from the previous plot
    plt.plot(time_axis, np.mean(sorted_data, axis=0), color="cyan", label="Mean Signal", linewidth=2)

    # Vertical line at event start (time=0)
    plt.axvline(0, color='blue', linestyle='--', linewidth=2, label="Start of Increasing")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load configuration from the JSON file
    with open("config.json", "r") as file:
        configs = json.load(file)

    # Iterate over the configurations and call extract_data for each
    for config in configs:
        extract_data(
            config["data_dir"],
            config["prefix"],
            config["before"],
            config["after"],
            config["split_minutes"]
        )