import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def process_data(data_dir, split_minutes, prefix):
    """
    Processes phyto node data to prepare it for training, including:
    - Merging datasets.
    - Analyzing and visualizing phase and heat data.
    - Generating time-sliced datasets.
    - Balancing the dataset for training.
    
    Args:
        data_dir (str): Path to the data directory.
        split_minutes (int): Time slice length in minutes.
        prefix (str): Prefix for the dataset (e.g., P3, P5).
    """

    # Define file paths
    temp_annotated_path = os.path.join(data_dir, "preprocessed/temp_annotated.csv")
    preprocessed_path = os.path.join(data_dir, f"preprocessed/{prefix}_preprocessed.csv")
    plants_path = os.path.join(data_dir, "plants.csv")

    # Load data
    temp_annotated = pd.read_csv(temp_annotated_path, parse_dates=['datetime'])
    preprocessed = pd.read_csv(preprocessed_path, parse_dates=['datetime'])
    plants = pd.read_csv(plants_path)

    # Merge datasets using nearest datetime
    merged_data = pd.merge_asof(
        preprocessed.sort_values('datetime'),
        temp_annotated.sort_values('datetime'),
        on='datetime',
        direction='nearest'
    )

    # Count and visualize datasets per phase
    phase_counts = merged_data['phase'].value_counts()

    plt.figure(figsize=(10, 6))
    ax = phase_counts.plot(kind='bar', color=['grey', 'red', 'purple', 'green'])
    plt.title(f"Number of Datasets per Phase from {prefix}")
    plt.xlabel("Phase")
    plt.ylabel("Number of Datasets")
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    for i, count in enumerate(phase_counts):
        ax.text(i, count + 0.02 * max(phase_counts), str(count), ha='center', fontsize=10)

    plt.show()

    # Plot data visualization (CH1, CH2, and Temperature with phases)
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    axes[0].plot(merged_data['datetime'], merged_data['CH1_smoothed_scaled'], color='black', label='CH1')
    axes[0].set_title(f'CH1 of {prefix}')
    axes[0].set_ylabel('Preprocessed Data')
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(merged_data['datetime'], merged_data['CH2_smoothed_scaled'], color='black', label='CH2')
    axes[1].set_title(f'CH2 of {prefix}')
    axes[1].set_ylabel('Preprocessed Data')
    axes[1].grid()
    axes[1].legend()

    for phase, color in [('Increasing', 'green'), ('Decreasing', 'red'), ('Holding', 'purple'), ('Nothing', 'grey')]:
        phase_data = merged_data[merged_data['phase'] == phase]
        axes[2].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
    
    axes[2].set_title('Temperature with Phases')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].grid()
    axes[2].legend(loc='upper right')

    plt.xlabel('Datetime')
    plt.tight_layout()
    plt.show()

    # Process data for output
    merged_data['Heat'] = merged_data['phase'].apply(lambda x: 1 if x in ['Increasing', 'Holding'] else 0)
    merged_data['Phase_Numbers'] = merged_data['phase'].map({'Nothing': 0, 'Decreasing': 1, 'Holding': 2, 'Increasing': 3})

    # Group data based on heat/phase changes
    merged_data['heat_group'] = (merged_data['Heat'].diff() != 0).cumsum()
    merged_data['phase_group'] = (merged_data['Phase_Numbers'].diff() != 0).cumsum()

    # Generate time-sliced data
    results = []
    for channel in ['CH1_smoothed_scaled', 'CH2_smoothed_scaled']:
        for group, group_data in merged_data.groupby(['heat_group']):
            current_time = group_data['datetime'].min()
            end_time = group_data['datetime'].max()
            while current_time + pd.Timedelta(minutes=split_minutes) <= end_time:
                slice_data = group_data[
                    (group_data['datetime'] >= current_time) &
                    (group_data['datetime'] < current_time + pd.Timedelta(minutes=split_minutes))
                ]
                if len(slice_data) == 60 * split_minutes and slice_data['Heat'].nunique() == 1:
                    row = {
                        'Start_Datetime': slice_data['datetime'].iloc[0],
                        'End_Datetime': slice_data['datetime'].iloc[-1],
                        'Plant': plants.iloc[0][f'{prefix}'],
                        'Channel': 1 if channel == 'CH1_smoothed_scaled' else 2,
                        'Phase': group_data['phase'].iloc[0],
                        'Heat': slice_data['Heat'].iloc[0]
                    }
                    row.update({f'val{i+1}': slice_data[channel].iloc[i] for i in range(len(slice_data))})
                    results.append(row)
                current_time += pd.Timedelta(minutes=split_minutes)

    # Create and balance the final DataFrame
    final_df = pd.DataFrame(results)

    heat_1 = final_df[final_df['Heat'] == 1] # upsampling possible --> smote function in case we need more data
    heat_0 = final_df[final_df['Heat'] == 0].sample(n=len(heat_1), random_state=42)
    final_df = pd.concat([heat_1, heat_0]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Visualize balanced dataset
    heat_counts = final_df['Heat'].value_counts()
    plt.figure(figsize=(10, 6))
    heat_counts.plot(kind='bar', color=['blue', 'red'])
    plt.title("Balanced Datasets")
    plt.xlabel("Heat")
    plt.ylabel("Number of Datasets for both Channels")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

    # Plot annotation validation with three subplots (CH1, CH2, and Temperature)
    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

    # Plot CH1 blocks in final_df
    for heat, color in [(1, 'red'), (0, 'blue')]:
        heat_data = final_df[(final_df['Heat'] == heat) & (final_df['Channel'] == 1)]  # Filter for CH1
        for _, row in heat_data.iterrows():
            # Plot the block
            axes[0].plot([row['Start_Datetime'], row['End_Datetime']], [row['Heat'], row['Heat']], color=color, linewidth=2)
            # Add a short vertical line at the end of the block aligned with the y-axis
            axes[0].vlines(
                x=row['End_Datetime'],
                ymin=row['Heat'] - 0.01,  # Short line centered on the block
                ymax=row['Heat'] + 0.01,
                color='black',
                linestyle='--',
                linewidth=0.5
            )
    axes[0].set_title('Heat Blocks in CH1')
    axes[0].set_ylabel('Heat')
    axes[0].legend(["Heat 1", "Heat 0"], loc='upper right')
    axes[0].grid()

    # Plot CH2 blocks in final_df
    for heat, color in [(1, 'red'), (0, 'blue')]:
        heat_data = final_df[(final_df['Heat'] == heat) & (final_df['Channel'] == 2)]  # Filter for CH2
        for _, row in heat_data.iterrows():
            # Plot the block
            axes[1].plot([row['Start_Datetime'], row['End_Datetime']], [row['Heat'], row['Heat']], color=color, linewidth=2)
            # Add a short vertical line at the end of the block aligned with the y-axis
            axes[1].vlines(
                x=row['End_Datetime'],
                ymin=row['Heat'] - 0.01,  # Short line centered on the block
                ymax=row['Heat'] + 0.01,
                color='black',
                linestyle='--',
                linewidth=0.5
            )
    axes[1].set_title('Heat Blocks in CH2')
    axes[1].set_ylabel('Heat')
    axes[1].legend(["Heat 1", "Heat 0"], loc='upper right')
    axes[1].grid()

    # Plot preprocessed temperatures with phases
    for phase, color in [('Increasing', 'green'), ('Decreasing', 'red'), ('Holding', 'purple'), ('Nothing', 'grey')]:
        phase_data = merged_data[merged_data['phase'] == phase]
        axes[2].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
    axes[2].set_title('Preprocessed Temperatures with Phases')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].legend(loc='upper right')
    axes[2].grid()

    plt.xlabel('Datetime')
    plt.tight_layout()
    plt.show()

    # Select a few random slices to plot
    slices_to_plot = final_df.sample(n=5) # Randomly select 5 slices

    # Plot each slice
    fig, axes = plt.subplots(len(slices_to_plot), 1, figsize=(12, len(slices_to_plot) * 3))

    for i, (_, slice_row) in enumerate(slices_to_plot.iterrows()):
        # Extract the channel data for this slice
        channel_data = [slice_row[f'val{j+1}'] for j in range(600)]
        
        # Plot the slice
        axes[i].plot(channel_data, label=f"Plant: {slice_row['Plant']}, Channel: {slice_row['Channel']}, Phase: {slice_row['Phase']}, Heat: {slice_row['Heat']}")
        axes[i].set_title(f"Slice {i + 1}")
        axes[i].set_xlabel("Time (seconds)")
        axes[i].set_ylabel("Scaled")
        axes[i].legend()
        axes[i].grid()

    plt.tight_layout()
    plt.show()



    # Save processed data
    output_dir = os.path.join(data_dir, "training_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prefix}_ready_to_train.csv")
    #final_df.drop(columns=['Start_Datetime', 'End_Datetime'], inplace=True)
    final_df.to_csv(output_path, index=False)

    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process phyto node data for training.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--split_minutes", type=int, default=10, help="Time slice length in minutes.")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix for the dataset (e.g., P3, P5).")
    args = parser.parse_args()

    process_data(args.data_dir, args.split_minutes, args.prefix)