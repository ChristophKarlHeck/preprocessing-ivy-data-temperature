import os
import pandas as pd
import glob

# Define constants
DATABITS = 8388608  # 23-bit ADC resolution
VREF = 2.5          # Reference voltage in volts
GAIN = 4.0 
WINDOW_SIZE = 5     # Rolling window size for smoothing
RESAMPLE_RATE = '1s'  # Resampling rate
DATA_DIR = "/home/chris/experiment_data/19.12.24-08.01.25" # containing P3, P5 and P6 (temperature data)
CUTOFF_DATE = "2024-12-20"
MIN_VALUE = -200 #mv for max min scaling
MAX_VALUE = 200 #mv

# Enable interactive mode for plotting (if needed)
import matplotlib.pyplot as plt
plt.ion()

# Directories
preprocessed_dir = os.path.join(DATA_DIR, "preprocessed")
os.makedirs(preprocessed_dir, exist_ok=True)

# Process P3 ivy data
# p3_files = glob.glob(os.path.join(DATA_DIR, "P3_*.csv"))
# p3_files_sorted = sorted(p3_files, key=lambda x: os.path.basename(x).split('_')[1])
# df_p3 = pd.concat([pd.read_csv(file) for file in p3_files_sorted], ignore_index=True)

# Process P5 ivy data
# p5_files = glob.glob(os.path.join(preprocessed_dir, "p5_*.csv"))
# p5_files_sorted = sorted(p5_files, key=lambda x: os.path.basename(x).split('_')[1])
# df_p5 = pd.concat([pd.read_csv(file) for file in p5_files_sorted], ignore_index=True)

# Process P6 (temperature) data
temp_files = glob.glob(os.path.join(DATA_DIR, "P6_*.csv"))
temp_files_sorted = sorted(temp_files, key=lambda x: os.path.basename(x).split('_')[1])
df_temp = pd.concat([pd.read_csv(file) for file in temp_files_sorted], ignore_index=True)
df_temp = df_temp.rename(columns={'timestamp': 'datetime'}) # rename column timestamp

# Convert datetime columns to pandas datetime format
# df_p3['datetime'] = pd.to_datetime(df_p3['datetime'], format='%Y-%m-%d %H:%M:%S:%f', errors='coerce')
#df_p5['datetime'] = pd.to_datetime(df_p5['datetime'], format='%Y-%m-%d %H:%M:%S:%f', errors='coerce')
df_temp['datetime'] = pd.to_datetime(df_temp['datetime'], format='%Y-%m-%d_%H:%M:%S:%f', errors='coerce')


# Drop rows with invalid datetime
# df_p3 = df_p3.dropna(subset=['datetime']) 
#df_p5 = df_p5.dropna(subset=['datetime']) 
df_temp = df_temp.dropna(subset=['datetime'])



# Set datetime as index
# df_p3.set_index('datetime', inplace=True)
#df_p5.set_index('datetime', inplace=True)
df_temp.set_index('datetime', inplace=True)


# Cut off data before cutoff_date
# df_p3 = df_p3[df_p3.index >= CUTOFF_DATE]
#df_p5 = df_p5[df_p5.index >= CUTOFF_DATE]
df_temp = df_temp[df_temp.index >= CUTOFF_DATE]


# Resample to 1 Hz and interpolate missing values
# df_p3_resampled = df_p3.resample(RESAMPLE_RATE).mean().interpolate()
#df_p5_resampled = df_p5.resample(RESAMPLE_RATE).mean().interpolate()
df_temp_resampled = df_temp.resample(RESAMPLE_RATE).mean().interpolate()

df_p5_resampled = pd.read_csv(os.path.join(preprocessed_dir, "p5_preprocessed.csv"), parse_dates=['datetime'], index_col='datetime')

print(df_p5_resampled.head())

# Convert to milli Volt
# df_p3_resampled['CH1_mv'] = ((df_p3_resampled['CH1'] / DATABITS - 1) * VREF / GAIN) * 1000  # Convert to millivolts
# df_p3_resampled['CH2_mv'] = ((df_p3_resampled['CH2'] / DATABITS - 1) * VREF / GAIN) * 1000  # Convert to millivolts

df_p5_resampled['CH1_mv'] = ((df_p5_resampled['CH1'] / DATABITS - 1) * VREF / GAIN) * 1000  # Convert to millivolts
df_p5_resampled['CH2_mv'] = ((df_p5_resampled['CH2'] / DATABITS - 1) * VREF / GAIN) * 1000  # Convert to millivolts

# Apply a simple moving average filter to CH1 and CH2
# Adjust the Window Size: Test different WINDOW_SIZE values (e.g., 3, 5, 10) to find a balance between noise reduction and preserving trends.
# df_p3_resampled["CH1_smooth"] = df_p3_resampled["CH1_mv"].rolling(window=WINDOW_SIZE).mean().dropna()
# df_p3_resampled["CH2_smooth"] = df_p3_resampled["CH2_mv"].rolling(window=WINDOW_SIZE).mean().dropna()

df_p5_resampled["CH1_smooth"] = df_p5_resampled["CH1_mv"].rolling(window=WINDOW_SIZE).mean()
df_p5_resampled["CH2_smooth"] = df_p5_resampled["CH2_mv"].rolling(window=WINDOW_SIZE).mean()

# Function to apply min-max scaling
def min_max_scale(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[f"{column}_scaled"] = (df[column] - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)

# Scale the data
# min_max_scale(df_p3_resampled, "CH1_smooth")
# min_max_scale(df_p3_resampled, "CH2_smooth")
min_max_scale(df_p5_resampled, "CH1_smooth")
min_max_scale(df_p5_resampled, "CH2_smooth")

# Write to CSV
# df_p3_resampled.to_csv(os.path.join(preprocessed_dir, "p3_preprocessed.csv"))
df_p5_resampled.to_csv(os.path.join(preprocessed_dir, "p5_preprocessed_2.csv"))
df_temp_resampled.to_csv(os.path.join(preprocessed_dir, "temp_preprocessed.csv"))

print("Preprocessing complete. Files saved in:", preprocessed_dir)


# Create a figure with subplots and shared x-axis
fig, axs = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

#P3 CH1
# axs[0].plot(df_p3_resampled.index, df_p3_resampled["CH1_mv"], label="Resampled CH1")
# axs[0].plot(df_p3_resampled.index, df_p3_resampled["CH1_smooth"], label="Smoothed CH1", linestyle="--")
# axs[0].set_title("P3 CH1 Resampled, Smoothed, and Scaled")
# axs[0].set_ylabel("Value")
# axs[0].legend()
# axs[0].grid()

#P3 CH2
# axs[1].plot(df_p3_resampled.index, df_p3_resampled["CH2_mv"], label="Resampled CH2")
# axs[1].plot(df_p3_resampled.index, df_p3_resampled["CH2_smooth"], label="Smoothed CH2", linestyle="--")
# axs[1].set_title("P3 CH2 Resampled, Smoothed, and Scaled")
# axs[1].set_ylabel("Value")
# axs[1].legend()
# axs[1].grid()

# #P5 CH1
axs[0].plot(df_p5_resampled.index, df_p5_resampled["CH1_mv"], label="Resampled CH1")
axs[0].plot(df_p5_resampled.index, df_p5_resampled["CH1_smooth"], label="Smoothed CH1", linestyle="--")
axs[0].set_title("P5 CH1: Resampled and Smoothed Data")
axs[0].set_ylabel("mV")
axs[0].legend()
axs[0].grid()

# # P5 CH2
axs[1].plot(df_p5_resampled.index, df_p5_resampled["CH2_mv"], label="Resampled CH2")
axs[1].plot(df_p5_resampled.index, df_p5_resampled["CH2_smooth"], label="Smoothed CH2", linestyle="--")
axs[1].set_title("P5 CH1: Resampled and Smoothed Data")
axs[1].set_ylabel("mV")
axs[1].legend()
axs[1].grid()

axs[2].plot(df_p5_resampled.index, df_p5_resampled["CH1_smooth_scaled"], label="Scaled CH1", linestyle=":")
axs[2].plot(df_p5_resampled.index, df_p5_resampled["CH2_smooth_scaled"], label="Scaled CH2", linestyle="--")
# axs[2].plot(df_p3_resampled.index, df_p3_resampled["CH1_smooth_scaled"], label="Scaled CH1", linestyle=":")
# axs[2].plot(df_p3_resampled.index, df_p3_resampled["CH2_smooth_scaled"], label="Scaled CH2", linestyle="--")
axs[2].set_title("P5 CH1,CH2: Resampled, Smoothed, and Scaled Data")
axs[2].set_ylabel("")
axs[2].legend()
axs[2].grid()

# Temperature Data
df_temp_resampled['avg_leaf_temp'] = (df_temp_resampled['T1_leaf'] + df_temp_resampled['T2_leaf']) / 2
df_temp_resampled['avg_air_temp'] = (df_temp_resampled['T1_air'] + df_temp_resampled['T2_air']) / 2
axs[3].plot(df_temp_resampled.index, df_temp_resampled['avg_leaf_temp'], label='Average Leaf Temperature', alpha=0.7)
axs[3].plot(df_temp_resampled.index, df_temp_resampled['avg_air_temp'], label='Average Air Temperature', alpha=0.7)
axs[3].set_title("Temperature Data (1 Hz)")
axs[3].set_xlabel("Datetime")
axs[3].set_ylabel("°C")
axs[3].legend()
axs[3].grid()

# Adjust layout
plt.tight_layout()

plot_path = os.path.join(preprocessed_dir, "p5_preprocessed_data.png")
plt.savefig(plot_path, dpi=300)

plt.show(block=True)


plt.close(fig)