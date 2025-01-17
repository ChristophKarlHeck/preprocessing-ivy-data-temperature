import os
import pandas as pd
import glob

# Define constants
DATABITS = 8388608  # 23-bit ADC resolution
VREF = 2.5          # Reference voltage in volts
GAIN = 4.0 
WINDOW_SIZE = 5     # Rolling window size for smoothing
RESAMPLE_RATE = '1s'  # Resampling rate
DATA_DIR = "/home/chris/experiment_data/19.12.24-08.01.25"  # containing P3, P5 and P6 (temperature data)
CUTOFF_DATE = "2024-12-20"
MIN_VALUE = -200  # mv for max min scaling
MAX_VALUE = 200   # mv

# Directories
preprocessed_dir = os.path.join(DATA_DIR, "preprocessed")
os.makedirs(preprocessed_dir, exist_ok=True)

# Output file
output_file_p5 = os.path.join(preprocessed_dir, "p5_preprocessed.csv")

# Process P5 ivy data in chunks
p5_files = glob.glob(os.path.join(DATA_DIR, "P5_*.csv"))
p5_files_sorted = sorted(p5_files, key=lambda x: os.path.basename(x).split('_')[1])

# Initialize an empty file to append data later
with open(output_file_p5, 'w') as f:
    f.write("datetime,CH1,CH2\n")

# Process files in chunks
for file in p5_files_sorted:
    chunk_iterator = pd.read_csv(file, chunksize=10000)
    
    for chunk in chunk_iterator:
        # Drop rows with invalid datetime
        chunk['datetime'] = pd.to_datetime(chunk['datetime'], format='%Y-%m-%d %H:%M:%S:%f', errors='coerce')
        chunk = chunk.dropna(subset=['datetime'])

        # Set datetime as index
        chunk.set_index('datetime', inplace=True)

        # Cut off data before cutoff_date
        chunk = chunk[chunk.index >= CUTOFF_DATE]

        # Resample to 1 Hz and interpolate missing values
        chunk_resampled = chunk.resample(RESAMPLE_RATE).mean().interpolate()

        # Append processed chunk to the output file
        chunk_resampled.to_csv(output_file_p5, mode='a', header=not os.path.exists(output_file_p5))

print(f"Processing complete. Combined file saved at: {output_file_p5}")