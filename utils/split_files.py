import pandas as pd
import os

DATA_DIR = "/home/chris/experiment_data/raw_19.12.24-08.01.25"

# Define the file path
input_file = os.path.join(DATA_DIR, "P5_2024-12-19_13:58:11:484588.csv")

# Define date ranges and output file names
ranges = {
    "P5_2024-12-20_to_2024-12-26.csv": ("2024-12-20", "2024-12-27"),
    "P5_2024-12-27_to_2025-01-03.csv": ("2024-12-27", "2025-01-04"),
    "P5_2025-01-04_to_2025-01-08.csv": ("2025-01-04", "2025-01-08")
}

# Define output directory
output_dir = "split_files"
preprocessed_dir = os.path.join(DATA_DIR, output_dir)
os.makedirs(preprocessed_dir, exist_ok=True)

# Process the file in chunks
chunk_size = 10000  # Adjust the chunk size based on your system's memory
for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    for filename, (start_date, end_date) in ranges.items():
        # Filter rows within the date range
        subset = chunk[(chunk['datetime'] >= start_date) & (chunk['datetime'] <= end_date)]
        
        # Append the filtered subset to the corresponding output file
        output_path = os.path.join(preprocessed_dir, filename)
        subset.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))

# Confirm completion
print(f"Files saved in {preprocessed_dir}")