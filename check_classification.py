"""
Author: Christoph Karl Heck
Date: 2025-01-20
Description: This script preprocesses CSV files for phyto and temperature node data. 
             It includes data filtering, resampling, smoothing, scaling, and visualization.
             The output includes preprocessed CSV files and visual plots.
"""

from datetime import datetime
from rich.console import Console
import os
import pandas as pd
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt
from typing import Optional
from scipy.interpolate import make_interp_spline

# Constants
CONFIG = {
    "DATABITS": 8388608,
    "VREF": 2.5,
    "GAIN": 4.0,
    "WINDOW_SIZE": 5,
    "RESAMPLE_RATE": "1s",
    "MIN_VALUE": -0.2,
    "MAX_VALUE": 0.2,
}

# Initialize the console
console = Console()


def validate_date(date_str: str) -> str:
    """
    Validate and format the date.
    Args:
        date_str (str): Input date string.
    Returns:
        str: Formatted date string in YYYY-MM-DD format.
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid cutoff_date format. Use 'YYYY-MM-DD'.")


def discover_files(data_dir: str, prefix: str) -> list[str]:
    """
    Discover CSV files matching the given prefix in the specified directory.
    Args:
        data_dir (str): Directory to search.
        prefix (str): File prefix to match.
    Returns:
        list[str]: List of matching file paths.
    """
    console.print(f"[bold cyan]Discovering files with prefix '{prefix}' in '{data_dir}'[/bold cyan]")
    files = glob.glob(os.path.join(data_dir, f"{prefix}_*.csv"))
    console.print(f"Found [bold yellow]{len(files)}[/bold yellow] matching files.")
    return files


def load_and_combine_csv(files: list[str]) -> pd.DataFrame:
    """
    Load and combine multiple CSV files into a single DataFrame.
    Args:
        files (list[str]): List of file paths.
    Returns:
        pd.DataFrame: Combined DataFrame.
    """
    console.print("[bold green]Loading and combining CSV files into a single DataFrame...[/bold green]")
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)


def preprocess_data(df: pd.DataFrame, from_date: str, until_date: str) -> pd.DataFrame:
    """
    Preprocess the data: filter by date, resample, and smooth.
    Args:
        df (pd.DataFrame): Input DataFrame.
        from_date (str): Date to filter rows before.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    console.print("[bold yellow]Preprocessing data: filtering and resampling...[/bold yellow]")

    # Add dates to CONFIG
    CONFIG["FROM_DATE"] = from_date
    CONFIG["UNTIL_DATE"] = until_date

    df = df.dropna(subset=["datetime"])
    df.set_index("datetime", inplace=True)
    df = df[df.index >= from_date]
    df = df[df.index < until_date]
    df_resampled = df.resample(CONFIG["RESAMPLE_RATE"]).mean().interpolate()
    return df_resampled


def scale_column(df: pd.DataFrame, column: str) -> None:
    """
    Scale a column using min-max scaling.
    Args:
        df (pd.DataFrame): DataFrame containing the column.
        column (str): Column to scale.
    """
    console.print(f"[bold magenta]Scaling column '{column}' using Min-Max Scaling...[/bold magenta]")
    df[f"{column}_scaled"] = (df[column] - CONFIG["MIN_VALUE"]) / (
        CONFIG["MAX_VALUE"] - CONFIG["MIN_VALUE"]
    )


def plot_data(df_classified: pd.DataFrame, df_temp: pd.DataFrame, prefix: str, save_dir: str) -> None:
    """
    Plot and save the processed data.
    Args:
        df_phyto (pd.DataFrame): Phyto node data.
        df_temp (pd.DataFrame): Temperature node data.
        prefix (str): Prefix for file naming.
        save_dir (str): Directory to save the plots.
    """

    # Define colors based on classification threshold (0.5 for Class 0)
    colors_ch0 = ['blue' if p > 0.5 else 'red' for p in df_classified['ClassificationCh0_0']]
    colors_ch1 = ['blue' if p > 0.5 else 'red' for p in df_classified['ClassificationCh1_0']]

    fig, axs = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    # Define base y-positions for slope lines
    y_ch0 = 0.55  # Close to classification Ch0
    y_ch1 = 0.45  # Close to classification Ch

    # Scatter plot for classification
    axs[0].scatter(df_classified['datetime'], [y_ch0] * len(df_classified), c=colors_ch0, label="Ch0", marker='o', s=50)
    axs[0].scatter(df_classified['datetime'], [y_ch1] * len(df_classified), c=colors_ch1, label="Ch1", marker='s', s=50)
    axs[0].set_yticks([y_ch0, y_ch1])
    axs[0].set_yticklabels(["Ch1", "Ch0"])
    axs[0].set_title("Classification Over Time")
    axs[0].legend()
    axs[0].grid(True, axis='x', linestyle='--', alpha=0.7)

    # Line plot for interpolated electric potential
    axs[1].plot(df_classified['datetime'], df_classified['LastVoltageCh0'], label="Scaled Voltage Ch0", color="blue")
    axs[1].plot(df_classified['datetime'], df_classified['LastVoltageCh1'], label="Scaled Voltage Ch1", color="green", linestyle="dashed")

    # Labels and Titles
    axs[1].set_ylabel("Electric Potential (mV)")
    axs[1].set_title("Input NN")
    axs[1].set_xlabel("Datetime")
    axs[1].legend()
    axs[1].grid(True, axis='x', linestyle='--', alpha=0.7)

    # Temperature
    axs[2].plot(df_temp.index, df_temp["avg_leaf_temp"], label="Avg Leaf Temp", alpha=0.7)
    axs[2].plot(df_temp.index, df_temp["avg_air_temp"], label="Avg Air Temp", alpha=0.7)
    axs[2].set_title("Temperature Data")
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{prefix}_classified_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()

def save_config_to_txt(configuration: dict, directory: str, prefix: str) -> None:
    """
    Save the global configuration dictionary to a .txt file in the specified directory
    with a filename based on the given prefix.
    Args:
        configuration (dict): The configuration dictionary to save.
        directory (str): The directory where the file will be saved.
        prefix (str): The prefix for the filename.
    """
    filename = os.path.join(directory, f"{prefix}_config_used_for_classifying.txt")

    try:
        # Write the configuration to the file
        with open(filename, "w") as file:
            for key, value in configuration.items():
                file.write(f"{key} = {value}\n")
        
        console.print(f"[bold green]Configuration successfully saved to '{filename}'[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Failed to save configuration to '{filename}': {e}[/bold red]")

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Preprocess CSV files.")
    parser.add_argument("--prefix", required=True, help="Phyto node prefix (e.g., 'P5').")
    parser.add_argument("--data_dir", required=True, help="Directory with raw files.")
    parser.add_argument("--from_date", required=True, help="Cutoff date (YYYY-MM-DD). Cut off data before that date.")
    parser.add_argument("--until_date", required=True, help="Cutoff date (YYYY-MM-DD). Cut off data after that date.")
    args = parser.parse_args()

    # Normalize and validate inputs
    prefix = args.prefix.upper()
    data_dir = args.data_dir.lower()
    from_date = validate_date(args.from_date)
    until_date = validate_date(args.until_date)

    # Print Input Parameters
    console.print(f"[bold cyan]Prefix:[/bold cyan] {prefix}")
    console.print(f"[bold green]Data Directory:[/bold green] {data_dir}")
    console.print(f"[bold yellow]From:[/bold yellow] {from_date}")
    console.print(f"[bold yellow]Until:[/bold yellow] {until_date}")

    # Process Phyto Node 
    # phyto_files = discover_files(data_dir, prefix)
    # df_phyto = load_and_combine_csv(phyto_files)
    # df_phyto['datetime'] = pd.to_datetime(df_phyto['datetime'], format='%Y-%m-%d %H:%M:%S:%f', errors='coerce')
    # df_phyto = preprocess_data(df_phyto, from_date, until_date)
    # df_phyto["CH1_milli_volt"] = ((df_phyto["CH1"] / CONFIG["DATABITS"] - 1) * CONFIG["VREF"] / CONFIG["GAIN"]) * 1000
    # df_phyto["CH2_milli_volt"] = ((df_phyto["CH2"] / CONFIG["DATABITS"] - 1) * CONFIG["VREF"] / CONFIG["GAIN"]) * 1000
    # df_phyto["CH1_smoothed"] = df_phyto["CH1_milli_volt"].rolling(CONFIG["WINDOW_SIZE"]).mean()
    # df_phyto["CH2_smoothed"] = df_phyto["CH2_milli_volt"].rolling(CONFIG["WINDOW_SIZE"]).mean()
    # scale_column(df_phyto, "CH1_smoothed")
    # scale_column(df_phyto, "CH2_smoothed")

    # Process Classified Data
    classified_files = discover_files(data_dir, "C1")
    df_classified = load_and_combine_csv(classified_files)
    df_classified['datetime'] = pd.to_datetime(df_classified['Datetime'], format='%Y-%m-%d %H:%M:%S:%f', errors='coerce')
    # Extract classification probabilities
    df_classified['ClassificationCh0_0'] = df_classified['ClassificationCh0'].str.extract(r'\[(\d+\.\d+),')[0].astype(float)
    df_classified['ClassificationCh1_0'] = df_classified['ClassificationCh1'].str.extract(r'\[(\d+\.\d+),')[0].astype(float)

    # Convert VoltagesCh0NotScaled and VoltagesCh1NotScaled from string to lists of floats
    df_classified['VoltagesCh0'] = df_classified['VoltagesCh0NotScaled'].apply(lambda x: np.array(eval(x)))
    df_classified['VoltagesCh1'] = df_classified['VoltagesCh1NotScaled'].apply(lambda x: np.array(eval(x)))

    # Extract the last voltage value for both channels
    df_classified['LastVoltageCh0'] = df_classified['VoltagesCh0'].apply(lambda x: x[-1])
    df_classified['LastVoltageCh1'] = df_classified['VoltagesCh1'].apply(lambda x: x[-1])
    scale_column(df_classified, "LastVoltageCh0")
    scale_column(df_classified, "LastVoltageCh1")

    df_classified = df_classified.sort_values(by="Datetime").drop_duplicates(subset="Datetime").reset_index(drop=True)

 
    # Process Temperature Node
    temp_files = discover_files(data_dir, "P6")
    df_temp = load_and_combine_csv(temp_files)
    df_temp = df_temp.rename(columns={'timestamp': 'datetime'})
    df_temp['datetime'] = pd.to_datetime(df_temp['datetime'], format='%Y-%m-%d_%H:%M:%S:%f', errors='coerce')
    df_temp = preprocess_data(df_temp, from_date, until_date)
    df_temp["avg_leaf_temp"] = (df_temp["T1_leaf"] + df_temp["T2_leaf"]) / 2
    df_temp["avg_air_temp"] = (df_temp["T1_air"] + df_temp["T2_air"]) / 2

    # Save and Plot
    preprocessed_dir = os.path.join(data_dir, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)
    save_config_to_txt(CONFIG, preprocessed_dir, prefix)
    plot_data(df_classified, df_temp, prefix, preprocessed_dir)


if __name__ == "__main__":
    main()
