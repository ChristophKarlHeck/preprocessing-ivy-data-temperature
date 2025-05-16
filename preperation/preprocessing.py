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
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional

# Constants
CONFIG = {
    "DATABITS": 8388608,
    "VREF": 2.5,
    "GAIN": 4.0,
    "WINDOW_SIZE": 6,
    "RESAMPLE_RATE": "1s",
    "MIN_VALUE": -200,
    "MAX_VALUE": 200,
    "FACTOR": 1,
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
    df[f"{column}_scaled"] = ((df[column] - CONFIG["MIN_VALUE"]) / (
        CONFIG["MAX_VALUE"]-CONFIG["MIN_VALUE"]) * CONFIG["FACTOR"]
    )

def plot_data(df_phyto: pd.DataFrame, df_temp: pd.DataFrame, prefix: str, save_dir: str) -> None:
    """
    Plot and save the processed data.
    Args:
        df_phyto (pd.DataFrame): Phyto node data.
        df_temp (pd.DataFrame): Temperature node data.
        prefix (str): Prefix for file naming.
        save_dir (str): Directory to save the plots.
    """
    fig_width = 5.90666
    fig, axs = plt.subplots(3, 1, figsize=(fig_width, 6), sharex=True)

    # Phyto node CH1
    #axs[0].plot(df_phyto.index, df_phyto["CH1_milli_volt"], label="CH1 Resampled", color="#FF0000")
    axs[0].plot(df_phyto.index, df_phyto["CH1_smoothed"], label="CH0 Resampled", color="#FF0000")
    #axs[0].set_title(f"CH0: Resampled and Smoothed")
    axs[0].set_ylabel("Voltage [mV]")  # Add y-axis label
    #axs[0].set_ylim(CONFIG["MIN_VALUE"], CONFIG["MAX_VALUE"])  # Fix y-axis range
    axs[0].legend(loc="upper left")
    axs[0].grid()

    # Phyto node CH2
    #axs[1].plot(df_phyto.index, df_phyto["CH2_milli_volt"], label="CH2 Resampled")
    axs[1].plot(df_phyto.index, df_phyto["CH2_smoothed"], label="CH1 Resampled", color="#8B0000")
    #axs[1].set_title(f"CH1: Resampled and Smoothed")
    axs[1].set_ylabel("Voltage [mV]")  # Add y-axis label
    #axs[1].set_ylim(CONFIG["MIN_VALUE"], CONFIG["MAX_VALUE"])  # Fix y-axis range
    axs[1].legend(loc="upper left")
    axs[1].grid()

    # Scaled Data
    # axs[2].plot(df_phyto.index, df_phyto["CH1_smoothed_scaled"], label="CH1 Scaled", linestyle=":")
    # axs[2].plot(df_phyto.index, df_phyto["CH2_smoothed_scaled"], label="CH2 Scaled", linestyle="--")
    # axs[2].set_title(f"{prefix} CH1, CH2: Scaled Data by factor {CONFIG['FACTOR']}")
    # axs[2].legend()
    # axs[2].grid()

    # Temperature
    axs[2].plot(df_temp.index, df_temp["avg_leaf_temp"], label="Avg Leaf Temp", alpha=0.7, color="#228B22")
    axs[2].plot(df_temp.index, df_temp["avg_air_temp"], label="Avg Air Temp", alpha=0.7, color= "#FF8C00")
    axs[2].set_title("Temperature Data")
    axs[2].set_ylabel("[°C]")
    axs[2].legend(loc="upper left")
    axs[2].grid()

    time_format = mdates.DateFormatter('%H:%M')

    for ax in axs:
        ax.xaxis.set_major_formatter(time_format)

    plt.tight_layout()
    plt.savefig(f"ExampleSignal.pgf", format="pgf", bbox_inches="tight", pad_inches=0.05)
    #plot_path = os.path.join(save_dir, f"{prefix}_plot.png")
    #plt.savefig(plot_path, dpi=300)
    #plt.show()

def save_config_to_txt(configuration: dict, directory: str, prefix: str) -> None:
    """
    Save the global configuration dictionary to a .txt file in the specified directory
    with a filename based on the given prefix.
    Args:
        configuration (dict): The configuration dictionary to save.
        directory (str): The directory where the file will be saved.
        prefix (str): The prefix for the filename.
    """
    filename = os.path.join(directory, f"{prefix}_config_used_for_preprocessing.txt")

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
    phyto_files = discover_files(data_dir, prefix)
    df_phyto = load_and_combine_csv(phyto_files)
    df_phyto['datetime'] = pd.to_datetime(df_phyto['datetime'], format='%Y-%m-%d %H:%M:%S:%f', errors='coerce')
    df_phyto = preprocess_data(df_phyto, from_date, until_date)
    df_phyto["CH1_milli_volt"] = ((df_phyto["CH1"] / CONFIG["DATABITS"] - 1) * CONFIG["VREF"] / CONFIG["GAIN"]) * 1000
    df_phyto["CH2_milli_volt"] = ((df_phyto["CH2"] / CONFIG["DATABITS"] - 1) * CONFIG["VREF"] / CONFIG["GAIN"]) * 1000
    df_phyto["CH1_smoothed"] = df_phyto["CH1_milli_volt"].rolling(CONFIG["WINDOW_SIZE"]).mean()
    df_phyto["CH2_smoothed"] = df_phyto["CH2_milli_volt"].rolling(CONFIG["WINDOW_SIZE"]).mean()
    scale_column(df_phyto, "CH1_smoothed")
    scale_column(df_phyto, "CH2_smoothed")
    #df_phyto["CH1_smoothed_scaled"] = df_phyto["CH1_smoothed"]*1000
    #df_phyto["CH2_smoothed_scaled"] = df_phyto["CH2_smoothed"]*1000

    # Process Temperature Node
    temp_files = discover_files(data_dir, "P6")
    df_temp = load_and_combine_csv(temp_files)
    df_temp = df_temp.rename(columns={'timestamp': 'datetime'})
    df_temp['datetime'] = pd.to_datetime(df_temp['datetime'], format='%Y-%m-%d_%H:%M:%S:%f', errors='coerce')
    df_temp = preprocess_data(df_temp, from_date, until_date)
    df_temp["avg_leaf_temp"] = (df_temp["T1_leaf"] + df_temp["T2_leaf"]) / 2
    df_temp["avg_air_temp"] = (df_temp["T1_air"] + df_temp["T2_air"]) / 2

    # Save and Plot
    preprocessed_dir = os.path.join(data_dir, "preprocessed_mm_1")
    #os.makedirs(preprocessed_dir, exist_ok=True)
    #df_phyto.to_csv(os.path.join(preprocessed_dir, f"{prefix}_preprocessed.csv"))
    #df_temp.to_csv(os.path.join(preprocessed_dir, "temperature_preprocessed.csv"))
    #save_config_to_txt(CONFIG, preprocessed_dir, prefix)
    plot_data(df_phyto, df_temp, prefix, preprocessed_dir)


if __name__ == "__main__":
    main()
