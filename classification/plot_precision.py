import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Create the argument parser and add arguments for data directory and file name
    parser = argparse.ArgumentParser(description="Plot precision curves from a CSV file")
    parser.add_argument("--data-dir", required=True, help="Directory where the CSV file is located")
    parser.add_argument("--file", required=True, help="CSV file name")
    args = parser.parse_args()
    
    # Construct the full file path
    file_path = os.path.join(args.data_dir, args.file)
    
    # Read the CSV file (adjust column names if needed)
    df = pd.read_csv(file_path)
    
    # Plotting the precision curves
    plt.figure(figsize=(10, 6))
    plt.plot(df['threshold'], df['precision_heat'], marker='o', label='Precision Heat')
    plt.plot(df['threshold'], df['recall_heat'], marker='o', label='Recall Heat')
    
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Precision vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
