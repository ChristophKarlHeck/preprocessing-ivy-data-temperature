import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#'CH1_smoothed' -> 'CH1_smoothed_scaled'

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
    min_val = data_slice.min()
    max_val = data_slice.max()
    range_val = max_val - min_val

    if range_val == 0:
        return np.zeros_like(data_slice)  # Avoid division by zero if all values are the same

    normalized = (data_slice - min_val) / range_val
    return normalized * factor

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
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot CH1
    axes[0].plot(merged_data['datetime'], merged_data['CH1_smoothed'], color='black', label='CH1')
    axes[0].set_title('CH1 Signal')
    axes[0].set_ylabel('Preprocessed Data')
    axes[0].grid()
    axes[0].legend()
    
    # Plot CH2
    axes[1].plot(merged_data['datetime'], merged_data['CH2_smoothed'], color='black', label='CH2')
    axes[1].set_title('CH2 Signal')
    axes[1].set_ylabel('Preprocessed Data')
    axes[1].grid()
    axes[1].legend()
    
    # Plot Temperature with Phase Colors
    for phase, color in [('Increasing', 'green'), ('Decreasing', 'red'), ('Holding', 'purple'), ('Nothing', 'grey')]:
        phase_data = merged_data[merged_data['phase'] == phase]
        axes[2].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
    
    axes[2].set_title('Temperature with Phases')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].grid()
    axes[2].legend(loc='upper right')
    
    # Plot vertical lines and highlight domain for distinct increasing phase starts
    for i, ax in enumerate(axes):
        for j, start_time in enumerate(distinct_increasing_starts):
            label = "Start of Increasing" if j == 0 else None
            ax.axvline(start_time, color='blue', linestyle='--', linewidth=1.5, label=label)
            
            # Highlighting domain (before and after)
            ax.axvspan(start_time - pd.Timedelta(minutes=before), start_time + pd.Timedelta(minutes=after), 
                       color='blue', alpha=0.2)
    
    # Add legend only once per plot
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    axes[2].legend(loc='upper right')
    
    plt.xlabel('Datetime')
    plt.tight_layout()
    
    # Display the plot
    plt.show()

    # Extract 60-minute segments around each increasing start
    time_window_before = pd.Timedelta(minutes=before)
    time_window_after = pd.Timedelta(minutes=after)
    segments_ch1, segments_ch2, segments_datetime = [], [], []

    for start_time in distinct_increasing_starts:
        mask = (merged_data['datetime'] >= start_time - time_window_before) & (merged_data['datetime'] <= start_time + time_window_after)
        segment = merged_data.loc[mask, ['datetime', 'CH1_smoothed', 'CH2_smoothed']]
        segment_ch1 = segment['CH1_smoothed'].values
        #segment_ch1 = min_max_normalize(segment_ch1,1000)
        #segment_ch1 = z_score_normalize(segment_ch1,1000)
        segments_ch1.append(segment_ch1)
        segment_ch2 = segment['CH2_smoothed'].values
        #segment_ch2 = min_max_normalize(segment_ch2,1000)
        #segment_ch2 = z_score_normalize(segment_ch2,1000)
        segments_ch2.append(segment_ch2)
        segments_datetime.append(segment['datetime'].values)

    
    num_slices = int((before+after)/split_minutes)  # Each segment is split into 6 slices (10 min each)
    samples_per_slice = int(((before+after)*60)/num_slices) # 10 minutes at 1Hz sampling

    results = []
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

    #print(f"Processed slices stored in {output_path}")

    # Convert to NumPy arrays for processing
    segments_ch1 = np.array(segments_ch1)
    segments_ch2 = np.array(segments_ch2)

    # Compute mean and standard deviation
    mean_ch1, std_ch1 = np.nanmean(segments_ch1, axis=0), np.nanstd(segments_ch1, axis=0)
    mean_ch2, std_ch2 = np.nanmean(segments_ch2, axis=0), np.nanstd(segments_ch2, axis=0)
    
    # Time axis for plotting
    time_axis = np.linspace(-before, after, segments_ch1.shape[1])
    
    # Plot mean and standard deviation as shaded regions
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, mean_ch1, color='black', label='CH1 Mean')
    plt.fill_between(time_axis, mean_ch1 - std_ch1, mean_ch1 + std_ch1, color='black', alpha=0.2)
    
    plt.plot(time_axis, mean_ch2, color='red', label='CH2 Mean')
    plt.fill_between(time_axis, mean_ch2 - std_ch2, mean_ch2 + std_ch2, color='red', alpha=0.2)
    
    plt.axvline(0, color='blue', linestyle='--', label='Start of Increasing')
    plt.xlabel("Time (minutes relative to Increasing start)")
    plt.ylabel("Signal Intensity")
    plt.title("CH1 and CH2 Response Around Increasing Phase Start")
    plt.legend()
    plt.grid()
    plt.show()

    # Convert Start_Datetime and End_Datetime to datetime
    df_results['Start_Datetime'] = pd.to_datetime(df_results['Start_Datetime'])
    df_results['End_Datetime'] = pd.to_datetime(df_results['End_Datetime'])

    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Plot CH1
    for heat, color in zip([0, 1], ['blue', 'orange']):
        subset = df_results[(df_results['Channel'] == 0) & (df_results['Heat'] == heat)]
        first_entry = True
        for _, row in subset.iterrows():
            time_range = pd.date_range(start=row['Start_Datetime'], end=row['End_Datetime'], periods=len(row)-5)
            values = row[[col for col in row.index if col.startswith('val_')]].values
            axes[0].plot(time_range, values, color=color, alpha=0.5, label=f'CH0 Heat {heat}' if first_entry else "")
            first_entry = False
    axes[0].set_title('CH0 Signal')
    axes[0].set_ylabel('Preprocessed Data')
    axes[0].grid()
    axes[0].legend()

    # Plot CH2
    for heat, color in zip([0, 1], ['blue', 'orange']):
        subset = df_results[(df_results['Channel'] == 1) & (df_results['Heat'] == heat)]
        first_entry = True
        for _, row in subset.iterrows():
            time_range = pd.date_range(start=row['Start_Datetime'], end=row['End_Datetime'], periods=len(row)-5)
            values = row[[col for col in row.index if col.startswith('val_')]].values
            axes[1].plot(time_range, values, color=color, alpha=0.5, label=f'CH1 Heat {heat}' if first_entry else "")
            first_entry = False
    axes[1].set_title('CH1 Signal')
    axes[1].set_ylabel('Preprocessed Data')
    axes[1].grid()
    axes[1].legend()

    # Plot Temperature with Phase Colors
    for phase, color in [('Increasing', 'green'), ('Decreasing', 'red'), ('Holding', 'purple'), ('Nothing', 'grey')]:
        phase_data = temp_annotated[temp_annotated['phase'] == phase]
        axes[2].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)

    axes[2].set_title('Temperature with Phases')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].grid()
    axes[2].legend(loc='upper right')

    plt.xlabel('Datetime')
    plt.tight_layout()
    plt.show()

    #--------------------------------------------Z-Scaling-----------------------------------------------------------------------------#

    # results_z_scaling = []
    # segment_ch1_z_scaling = []
    # segment_ch2_z_scaling = []
    # for segment_index, (segment_ch1, segment_ch2, segment_datetime) in enumerate(zip(segments_ch1, segments_ch2, segments_datetime)):
    #     plant_name = plants.iloc[0][f'{prefix}']  # Extract plant name

    #     local_ch1_z_scaling=[]
    #     local_ch2_z_scaling=[]
    #     for slice_index in range(num_slices):
    #         start_idx = slice_index * samples_per_slice
    #         end_idx = start_idx + samples_per_slice
            
    #         # Extract the 10-min slice for both CH1 and CH2
            

    #         # Determine the Heat label
    #         heat = 0 if slice_index < int((before/split_minutes)) else 1  # First 3 slices -> 0, Last 3 slices -> 1

    #         # Construct row with 600 data points for both channels + metadata
    #         slice_data_ch1 = segment_ch1[start_idx:end_idx]
    #         slice_data_ch1_scaled = z_score_normalize(slice_data_ch1, 1000)
    #         local_ch1_z_scaling.extend(slice_data_ch1_scaled)
    #         row_ch0 = {
    #             'Start_Datetime': segment_datetime[start_idx],
    #             'End_Datetime': segment_datetime[end_idx],
    #             'Plant': plant_name,
    #             'Channel': 0,
    #             'Heat': heat
    #         }
    #         row_ch0.update({f'val_{i}': slice_data_ch1_scaled[i] for i in range(samples_per_slice)})  # CH1 values
    #         results_z_scaling.append(row_ch0)

    #         slice_data_ch2 = segment_ch2[start_idx:end_idx]
    #         slice_data_ch2_scaled = z_score_normalize(slice_data_ch2, 1000)
    #         local_ch2_z_scaling.extend(slice_data_ch2_scaled)
    #         row_ch1 = {
    #             'Start_Datetime': segment_datetime[start_idx],
    #             'End_Datetime': segment_datetime[end_idx],
    #             'Plant': plant_name,
    #             'Channel': 1,
    #             'Heat': heat
    #         }
    #         row_ch1.update({f'val_{i}': slice_data_ch2_scaled[i] for i in range(samples_per_slice)})  # CH2 values

    #         results_z_scaling.append(row_ch1)
    #     segment_ch1_z_scaling.append(local_ch1_z_scaling)
    #     segment_ch2_z_scaling.append(local_ch2_z_scaling)

    
    # df_results_z_scaling = pd.DataFrame(results_z_scaling)
    # output_dir = os.path.join(data_dir, "training_data")
    # os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, f"{prefix}_ready_to_train.csv")
    # df_results_z_scaling.to_csv(output_path, index=False)

    # # Convert to NumPy arrays for processing
    # segments_ch1 = np.array(segment_ch1_z_scaling)
    # segments_ch2 = np.array(segment_ch2_z_scaling)

    # # Compute mean and standard deviation
    # mean_ch1, std_ch1 = np.nanmean(segments_ch1, axis=0), np.nanstd(segments_ch1, axis=0)
    # mean_ch2, std_ch2 = np.nanmean(segments_ch2, axis=0), np.nanstd(segments_ch2, axis=0)
    
    # # Time axis for plotting
    # time_axis = np.linspace(-before, after, segments_ch1.shape[1])
    
    # # Plot mean and standard deviation as shaded regions
    # plt.figure(figsize=(10, 6))
    # plt.plot(time_axis, mean_ch1, color='black', label='CH1 Mean')
    # plt.fill_between(time_axis, mean_ch1 - std_ch1, mean_ch1 + std_ch1, color='black', alpha=0.2)
    
    # plt.plot(time_axis, mean_ch2, color='red', label='CH2 Mean')
    # plt.fill_between(time_axis, mean_ch2 - std_ch2, mean_ch2 + std_ch2, color='red', alpha=0.2)
    
    # plt.axvline(0, color='blue', linestyle='--', label='Start of Increasing')
    # plt.xlabel("Time (minutes relative to Increasing start)")
    # plt.ylabel("Signal Intensity")
    # plt.title("CH1 and CH2 Response Around Increasing Phase Start")
    # plt.legend()
    # plt.grid()

    # # Convert Start_Datetime and End_Datetime to datetime
    # df_results_z_scaling['Start_Datetime'] = pd.to_datetime(df_results_z_scaling['Start_Datetime'])
    # df_results_z_scaling['End_Datetime'] = pd.to_datetime(df_results_z_scaling['End_Datetime'])

    # # Create plots
    # fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # # Plot CH1
    # for heat, color in zip([0, 1], ['blue', 'orange']):
    #     subset = df_results_z_scaling[(df_results_z_scaling['Channel'] == 0) & (df_results_z_scaling['Heat'] == heat)]
    #     first_entry = True
    #     for _, row in subset.iterrows():
    #         time_range = pd.date_range(start=row['Start_Datetime'], end=row['End_Datetime'], periods=len(row)-5)
    #         values = row[[col for col in row.index if col.startswith('val_')]].values
    #         axes[0].plot(time_range, values, color=color, alpha=0.5, label=f'CH0 Heat {heat}' if first_entry else "")
    #         first_entry = False
    # axes[0].set_title('CH0 Signal')
    # axes[0].set_ylabel('Preprocessed Data')
    # axes[0].grid()
    # axes[0].legend()

    # # Plot CH2
    # for heat, color in zip([0, 1], ['blue', 'orange']):
    #     subset = df_results_z_scaling[(df_results_z_scaling['Channel'] == 1) & (df_results_z_scaling['Heat'] == heat)]
    #     first_entry = True
    #     for _, row in subset.iterrows():
    #         time_range = pd.date_range(start=row['Start_Datetime'], end=row['End_Datetime'], periods=len(row)-5)
    #         values = row[[col for col in row.index if col.startswith('val_')]].values
    #         axes[1].plot(time_range, values, color=color, alpha=0.5, label=f'CH1 Heat {heat}' if first_entry else "")
    #         first_entry = False
    # axes[1].set_title('CH1 Signal')
    # axes[1].set_ylabel('Preprocessed Data')
    # axes[1].grid()
    # axes[1].legend()

    # # Plot Temperature with Phase Colors
    # for phase, color in [('Increasing', 'green'), ('Decreasing', 'red'), ('Holding', 'purple'), ('Nothing', 'grey')]:
    #     phase_data = temp_annotated[temp_annotated['phase'] == phase]
    #     axes[2].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)

    # axes[2].set_title('Temperature with Phases')
    # axes[2].set_ylabel('Temperature (°C)')
    # axes[2].grid()
    # axes[2].legend(loc='upper right')

    # plt.xlabel('Datetime')
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process phyto node data for training.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix for the dataset (e.g., P3, P5).")
    parser.add_argument("--before", type=int, default=30, help="Time in minutes befor increase starts.")
    parser.add_argument("--after", type=int, default=30, help="Time in minutes after increase starts.")
    parser.add_argument("--split_minutes", type=int, default=10, help="Time slice length in minutes.")
    args = parser.parse_args()

    extract_data(args.data_dir, args.prefix, args.before, args.after, args.split_minutes)
