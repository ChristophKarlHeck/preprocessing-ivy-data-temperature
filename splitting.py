import pandas as pd
import matplotlib.pyplot as plt

# Reload the files after the reset
temp_annotated_path = "/home/chris/experiment_data/5_09.01.25-15.01.25/preprocessed/temp_annotated.csv"
p3_preprocessed_path = "/home/chris/experiment_data/5_09.01.25-15.01.25/preprocessed/P3_preprocessed.csv"
plants_path = "/home/chris/experiment_data/5_09.01.25-15.01.25/plants.csv"

temp_annotated = pd.read_csv(temp_annotated_path, parse_dates=['datetime'])
p3_preprocessed = pd.read_csv(p3_preprocessed_path, parse_dates=['datetime'])
print(p3_preprocessed.describe())
plants = pd.read_csv(plants_path)

# Merge temp_annotated with P3_Preprocessed
merged_data = pd.merge_asof(
    p3_preprocessed.sort_values('datetime'),
    temp_annotated.sort_values('datetime'),
    on='datetime',
    direction='nearest'
)

# Count the number of datasets from each phase
phase_counts = merged_data['phase'].value_counts()

# Plot the dataset counts for each phase
plt.figure(figsize=(10, 6))
phase_counts.plot(kind='bar', color=['green', 'red', 'purple', 'grey'])
plt.title("Number of Datasets per Phase")
plt.xlabel("Phase")
plt.ylabel("Number of Datasets")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Plot the three subplots
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# Plot CH1
axes[0].plot(merged_data['datetime'], merged_data['CH1_milli_volt'], color='blue', label='CH1')
axes[0].set_title('CH1 of P3')
axes[0].set_ylabel('CH1')
axes[0].grid()
axes[0].legend()

# Plot CH2
axes[1].plot(merged_data['datetime'], merged_data['CH2_milli_volt'], color='green', label='CH2')
axes[1].set_title('CH2 of P3')
axes[1].set_ylabel('CH2')
axes[1].grid()
axes[1].legend()

# Plot Temperature with Phases
for phase, color in [('Increasing', 'green'), ('Decreasing', 'red'), ('Holding', 'purple'), ('Nothing', 'grey')]:
    phase_data = merged_data[merged_data['phase'] == phase]
    axes[2].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
axes[2].set_title('Temperature with Phases')
axes[2].set_ylabel('Temperature (\u00b0C)')
axes[2].grid()
axes[2].legend(loc='upper right')

plt.xlabel('Datetime')
plt.tight_layout()
plt.show()

# Process data for writing the new file

# Create a column to identify continuous datetime blocks
merged_data['time_diff'] = merged_data['datetime'].diff().gt(pd.Timedelta(seconds=1))
merged_data['group'] = merged_data['time_diff'].cumsum()


# Drop the auxiliary column as it's no longer needed
merged_data.drop(columns=['time_diff'], inplace=True)

merged_data['Heat'] = merged_data['phase'].apply(lambda x: 1 if x in ['Increasing', 'Holding'] else 0)

# Plot the three subplots
# Plot the two subplots
fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# First subplot: Heat annotation
for heat, color in [(1, 'red'), (0, 'blue')]:
    heat_data = merged_data[merged_data['Heat'] == heat]
    axes[0].scatter(heat_data['datetime'], heat_data['Heat'], color=color, label=f'Heat {heat}', s=10)
axes[0].set_title('Heat Annotation in merged_data')
axes[0].set_ylabel('Heat')
axes[0].legend()
axes[0].grid()

# Second subplot: Preprocessed temperatures with phases
for phase, color in [('Increasing', 'green'), ('Decreasing', 'red'), ('Holding', 'purple'), ('Nothing', 'grey')]:
    phase_data = merged_data[merged_data['phase'] == phase]
    axes[1].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
axes[1].set_title('Preprocessed Temperatures with Phases')
axes[1].set_ylabel('Temperature (°C)')
axes[1].legend(loc='upper right')
axes[1].grid()

plt.xlabel('Datetime')
plt.tight_layout()
plt.show()


# Extract 10-minute slices based on continuous datetime groups
results = []
for channel in ['CH1_milli_volt', 'CH2_milli_volt']:
    for (phase, group), phase_data in merged_data.groupby(['Heat', 'group']):
        current_time = phase_data['datetime'].min()
        end_time = phase_data['datetime'].max()
        while current_time + pd.Timedelta(minutes=10) <= end_time:
            slice_data = phase_data[
                (phase_data['datetime'] >= current_time) &
                (phase_data['datetime'] < current_time + pd.Timedelta(minutes=10))
            ]
            if len(slice_data) == 600:# and slice_data['Heat'].nunique() == 1:
                row = {
                    'Start_Datetime': slice_data['datetime'].iloc[0],
                    'End_Datetime': slice_data['datetime'].iloc[-1],
                    'Plant': plants.iloc[0]['P3'],
                    'Channel': 1 if channel == 'CH1_milli_volt' else 2,
                    'Phase': phase,
                    'Heat': slice_data['Heat'].iloc[0]
                }
                row.update({f'val{i+1}': slice_data[channel].iloc[i] for i in range(600)})
                results.append(row)
            current_time += pd.Timedelta(minutes=10)

# Create the final DataFrame and save to CSV
final_df = pd.DataFrame(results)

# Filter out 'Decreasing' phase
final_df = final_df[final_df['Phase'] != 'Decreasing']

# Balance the dataset
heat_1 = final_df[final_df['Heat'] == 1]
heat_0 = final_df[final_df['Heat'] == 0].sample(n=len(heat_1), random_state=42)
final_df = pd.concat([heat_1, heat_0]).sample(frac=1, random_state=42).reset_index(drop=True)

heat_counts = final_df['Heat'].value_counts()
plt.figure(figsize=(10, 6))
heat_counts.plot(kind='bar', color=['blue', 'red'])
plt.title("Balanced Datasets")
plt.xlabel("Heat")
plt.ylabel("Number of Datasets")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

output_path = "/home/chris/experiment_data/5_09.01.25-15.01.25/preprocessed/P3_ready_to_train.csv"
final_df.to_csv(output_path, index=False)

# Plot annotation validation with two subplots
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot blocks in final_df
# Plot blocks in final_df
for heat, color in [(1, 'red'), (0, 'blue')]:
    heat_data = final_df[final_df['Heat'] == heat]
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
axes[0].set_title('Heat Blocks in final_df')
axes[0].set_ylabel('Heat')
axes[0].legend(["Heat 1", "Heat 0"], loc='upper right')
axes[0].grid()

# Plot preprocessed temperatures with phases
for phase, color in [('Increasing', 'green'), ('Decreasing', 'red'), ('Holding', 'purple'), ('Nothing', 'grey')]:
    phase_data = merged_data[merged_data['phase'] == phase]
    axes[1].scatter(phase_data['datetime'], phase_data['avg_air_temp'], color=color, label=phase, s=10)
axes[1].set_title('Preprocessed Temperatures with Phases')
axes[1].set_ylabel('Temperature (°C)')
axes[1].legend(loc='upper right')
axes[1].grid()

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
    axes[i].set_ylabel("Milli-volt")
    axes[i].legend()
    axes[i].grid()

plt.tight_layout()
plt.show()