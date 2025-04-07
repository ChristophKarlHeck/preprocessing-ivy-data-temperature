import argparse
import os
import pandas as pd

def find_all_files(data_dir: str):

    # List to store file paths
    all_files = []

    # Walk through the directory recursively
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Create the full file path by joining the root and file name
            file_path = os.path.join(root, file)
            # Convert the relative file path to an absolute file path
            abs_file_path = os.path.abspath(file_path)
            all_files.append(abs_file_path)

    # Print all discovered files
    print("All files found in the directory:")
    for file_path in all_files:
        print(file_path)

    return all_files

def transform(file: str, data_dir: str):

    df = pd.read_csv(file)
    # Define the columns containing the 100 values
    val_cols = [f'val_{i}' for i in range(100)]

    # Create a new column that aggregates the 100 values into a list for each row
    df['values_array'] = df[val_cols].values.tolist()

    # Split the DataFrame into two based on the 'Channel' value
    df_ch0 = df[df['Channel'] == 0][['Datetime', 'Heat', 'values_array']].copy()
    df_ch1 = df[df['Channel'] == 1][['Datetime', 'Heat', 'values_array']].copy()

    # Rename the 'values_array' column in each DataFrame to reflect the channel
    df_ch0 = df_ch0.rename(columns={'values_array': 'input_not_normalized_ch0'})
    df_ch1 = df_ch1.rename(columns={'values_array': 'input_not_normalized_ch1'})

    # Optionally, check that the 'Heat' values are the same for both channels per datetime.
    # For simplicity, we'll assume they are and use the one from channel 0.

    # Merge the two DataFrames on 'Datetime'
    df_merged = pd.merge(df_ch0, df_ch1, on='Datetime', suffixes=('_ch0', '_ch1'))

    df_merged = df_merged.drop(columns=['Heat_ch1'])

    # Rename columns: 'Datetime' to 'datetime' and 'Heat' to 'ground_truth'
    df_merged = df_merged.rename(columns={'Datetime': 'datetime', 'Heat_ch0': 'ground_truth'})

    # Select and reorder the final columns
    df_final = df_merged[['datetime', 'ground_truth', 'input_not_normalized_ch0', 'input_not_normalized_ch1']]

    classification_dir = os.path.join(data_dir, "classification")
    os.makedirs(classification_dir, exist_ok=True)

    # Define the CSV file path inside the classification folder and save the final DataFrame as a CSV file
    basename = os.path.basename(file)
    csv_path = os.path.join(classification_dir, basename)
    df_final.to_csv(csv_path, index=False)
    print(f"CSV saved at: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process phyto node data for training.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory.")
    args = parser.parse_args()

    files = find_all_files(args.data_dir)

    for file in files:
        transform(file, args.data_dir)