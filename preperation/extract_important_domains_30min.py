import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def downsample_by_mean(data):
    data = np.array(data)
    if data.shape[0] % 6 != 0:
        raise ValueError("Length of data must be a multiple of 6.")

    # Reshape the data into a 2D array with each row containing 6 values.
    reshaped_data = data.reshape(-1, 6)
    
    # Compute the mean along the axis 1 (i.e. for each row)
    downsampled = reshaped_data.mean(axis=1)
    return downsampled

def downsample6to1(data):
    data = np.array(data)
    if data.size != 6:
        raise ValueError("Input data must contain exactly 6 values.")
    
    return data.mean()

def extract_data(data_dir, prefix, before, after):
    # Define file paths
    temp_annotated_path = os.path.join(data_dir, "preprocessed/temp_annotated.csv")
    preprocessed_path = os.path.join(data_dir, f"preprocessed_amm/{prefix}_preprocessed.csv") # CHANGE HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
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
    axes[0].plot(merged_data['datetime'], merged_data['CH1_smoothed_scaled'], color='black', label='CH1')
    axes[0].set_title('CH1 Signal')
    axes[0].set_ylabel('Preprocessed Data')
    axes[0].grid()
    axes[0].legend()
    
    # Plot CH2
    axes[1].plot(merged_data['datetime'], merged_data['CH2_smoothed_scaled'], color='black', label='CH2')
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
    time_window_before = pd.Timedelta(minutes=90)
    time_window_after = pd.Timedelta(minutes=60)

    results = []
    for start_time in distinct_increasing_starts:
        mask = (merged_data['datetime'] >= start_time - time_window_before) & (merged_data['datetime'] <= start_time + time_window_after)
        segment = merged_data.loc[mask, ['datetime', 'CH1_smoothed_scaled', 'CH2_smoothed_scaled', "phase"]]

        input_ch0 = downsample_by_mean(segment['CH1_smoothed_scaled'][0:1800].values)
        input_ch0 = z_score_normalize(input_ch0, 1000)
        init_datetime = segment['datetime'].iloc[1799]
        row_ch0 = {
                'Channel': 0,
                'Heat': 0,
                'Datetime': init_datetime
            }
        row_ch0.update({f'val_{z}': input_ch0[z] for z in range(len(input_ch0))})
        results.append(row_ch0)

        input_ch1 = downsample_by_mean(segment['CH2_smoothed_scaled'][0:1800].values)
        input_ch1 = z_score_normalize(input_ch1, 1000)
        row_ch1 = {
                'Channel': 1,
                'Heat': 0,
                'Datetime': init_datetime
            }
        row_ch1.update({f'val_{z}': input_ch1[z] for z in range(len(input_ch1))})
        results.append(row_ch1)

        segment_after_1800 = segment.iloc[1800:]
        n_groups = len(segment_after_1800) // 6  # Only process complete groups of 6 rows.
        
        for i in range(n_groups):
            group = segment_after_1800.iloc[i*6:(i+1)*6]
            heat_flag = 1 if group['phase'].isin(['Increasing', 'Holding']).any() else 0
            mean_ch0 = group['CH1_smoothed_scaled'].mean()
            mean_ch1 = group['CH2_smoothed_scaled'].mean()

            avg_ns = group['datetime'].astype(np.int64).mean()
            avg_datetime = pd.to_datetime(avg_ns)

            input_ch0 = np.append(input_ch0[1:], mean_ch0)
            input_ch1 = np.append(input_ch1[1:], mean_ch1)
            input_ch0 = z_score_normalize(input_ch0, 1000)
            input_ch1 = z_score_normalize(input_ch1, 1000)

            row_ch0 = {
                'Channel': 0,
                'Heat': heat_flag,
                'Datetime': avg_datetime
            }
            row_ch0.update({f'val_{j}': input_ch0[j] for j in range(len(input_ch0))})
            results.append(row_ch0)
            
            row_ch1 = {
                'Channel': 1,
                'Heat': heat_flag,
                'Datetime': avg_datetime
            }
            row_ch1.update({f'val_{j}': input_ch1[j] for j in range(len(input_ch1))})
            results.append(row_ch1)
            
    df_results = pd.DataFrame(results)

   # For plotting, you may want to focus on one channel (say, Channel 0).
    df_channel0 = df_results[df_results['Channel'] == 0]

    # Create a scatter plot showing the heat classification over time.
    plt.figure(figsize=(12, 6))
    # We'll plot 1 for heat and 0 for not heat; using different colors.
    plt.scatter(df_channel0['Datetime'], df_channel0['Heat'], 
                c=df_channel0['Heat'], cmap='coolwarm', marker='o')
    plt.xlabel("Datetime")
    plt.ylabel("Heat Classification (0 = Not Heat, 1 = Heat)")
    plt.title("Heat Classification Over Time (Channel 0)")
    plt.grid(True)
    plt.show()

    # Separate the two classes.
    df_heat0 = df_results[df_results['Heat'] == 0]
    df_heat1 = df_results[df_results['Heat'] == 1]

    # Determine the size of the minority class.
    n_min = min(len(df_heat0), len(df_heat1))

    # Downsample both classes to have the same number of rows.
    df_heat0_balanced = df_heat0.sample(n=n_min, random_state=42)
    df_heat1_balanced = df_heat1.sample(n=n_min, random_state=42)

    # Combine the balanced classes.
    df_balanced = pd.concat([df_heat0_balanced, df_heat1_balanced]).sort_values('Datetime').reset_index(drop=True)

    counts_original = df_results['Heat'].value_counts().sort_index()
    # Count classes in the balanced dataset
    counts_balanced = df_balanced['Heat'].value_counts().sort_index()

    # Create a figure with two subplots side by side.
    plt.figure(figsize=(12, 5))

    # Plot the original distribution.
    plt.subplot(1, 2, 1)
    plt.bar(counts_original.index, counts_original.values, tick_label=["Heat=0", "Heat=1"])
    plt.title("Original Class Distribution")
    plt.xlabel("Heat Class")
    plt.ylabel("Count")

    # Plot the balanced distribution.
    plt.subplot(1, 2, 2)
    plt.bar(counts_balanced.index, counts_balanced.values, tick_label=["Heat=0", "Heat=1"])
    plt.title("Balanced Class Distribution")
    plt.xlabel("Heat Class")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

    # Convert to DataFrame and save to CSV
    output_dir = os.path.join(data_dir, "training_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prefix}_ready_to_train.csv")
    df_balanced.to_csv(output_path, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process phyto node data for training.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix for the dataset (e.g., P3, P5).")
    parser.add_argument("--before", type=int, default=30, help="Time in minutes befor increase starts.")
    parser.add_argument("--after", type=int, default=30, help="Time in minutes after increase starts.")
    args = parser.parse_args()

    extract_data(args.data_dir, args.prefix, args.before, args.after)
