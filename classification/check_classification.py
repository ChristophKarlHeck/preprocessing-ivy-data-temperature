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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional
from scipy.interpolate import make_interp_spline

# Use the PGF backend
matplotlib.use("pgf")

# Update rcParams
plt.rcParams.update({
    "pgf.texsystem": "xelatex",  # Use XeLaTeX
    "font.family": "sans-serif",  # Use a sans-serif font
    "font.sans-serif": ["Arial"],  # Specifically use Arial
    "font.size": 10,  # Set the font size
    "text.usetex": True,  # Use LaTeX for text rendering
    "pgf.rcfonts": False,  # Do not override Matplotlib's rc settings
})

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
    Validate and format the datetime.
    Args:
        datetime_str (str): Input datetime string.
    Returns:
        str: Formatted datetime string in 'YYYY-MM-DD HH:MM' format.
    """
    try:
        # Parse the input string to a datetime object
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        # Return the formatted datetime string
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        raise ValueError("Invalid datetime format. Use 'YYYY-MM-DD HH:MM'.")


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


def cut_data(df: pd.DataFrame, from_date: str, until_date: str) -> pd.DataFrame:
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
    df = df[(df["datetime"] >= from_date) & (df["datetime"] < until_date)]
    return df

def resample_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("datetime", drop=False)
    df_resampled = df.resample(CONFIG["RESAMPLE_RATE"]).mean().interpolate()
    return df_resampled

def resample_input_nn_data(df: pd.DataFrame) -> None:
    """Resamples the DataFrame in place using the specified resample rate."""
    

    df = df.resample("1s").mean().interpolate()

def min_max_scale_column(df: pd.DataFrame, column: str) -> None:
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

def plot_data(df_classified: pd.DataFrame, df_input: pd.DataFrame, df_merged: pd.DataFrame, df_temp: pd.DataFrame, prefix: str, threshold: float, save_dir: str) -> None:
    """
    Plot and save the processed data.
    Args:
        df_phyto (pd.DataFrame): Phyto node data.
        df_temp (pd.DataFrame): Temperature node data.
        prefix (str): Prefix for file naming.
        save_dir (str): Directory to save the plots.
    """
    df_classified['datetime'] = df_classified['datetime'] + pd.Timedelta(hours=1)
    df_merged['datetime'] = df_merged['datetime'] + pd.Timedelta(hours=1)
    df_input['datetime'] = df_input['datetime'] + pd.Timedelta(hours=1)
    df_temp.index = df_temp.index + pd.Timedelta(hours=1)

    fig_width = 5.90666  # Width in inches
    aspect_ratio = 0.618  # Example aspect ratio (height/width)
    fig_height = fig_width * aspect_ratio

    fig, axs = plt.subplots(3, 1, figsize=(fig_width, 8), sharex=True)

    time_fmt = mdates.DateFormatter('%H:%M')

    for ax in axs:
        ax.grid(True, linestyle='dashed', linewidth=0.5, alpha=0.6)
        ax.xaxis.set_major_formatter(time_fmt)  # Format x-axis as hours
        ax.tick_params(axis='x', labelsize=10)  # Set font size to 10
        plt.setp(ax.get_xticklabels(), fontsize=10, rotation=0, ha='center')

    # Scatter plot for classification
    axs[0].plot(df_classified['datetime'], df_classified["ch0_smoothed"], label="CH0", color="blue")
    axs[0].plot(df_classified['datetime'], df_classified["ch1_smoothed"], label="CH1", color="green")

    axs[0].axhline(y=0.9, color="red", linestyle="--", linewidth=1, label="Threshold: 0.9")

    axs[0].fill_between(df_classified['datetime'], 0, 1.0, 
                    where=(df_classified["ch0_smoothed"] > threshold) & (df_classified["ch1_smoothed"] > threshold), 
                    color='gray', alpha=0.3, label="Stimulus prediction")


    axs[0].fill_between(df_merged['datetime'], 0, 1.0, 
                where=(df_merged["phase"].isin(["Increasing", "Holding"])), 
                    color='limegreen', alpha=0.3, label="Stimulus application")

    # Ensure y-axis limits and set explicit tick marks
    axs[0].set_ylim(0, 1.05)
    axs[0].set_yticks([0, 0.25, 0.5, 0.75, 1])  # Explicitly set y-ticks
    axs[0].set_ylabel("Heat Phase Probability",fontsize=10)
    axs[0].tick_params(axis='y', labelsize=10) 

    axs[0].set_title("Online Heat Phase Classification Using Ivy Data (ID 525)",fontsize=10)
    axs[0].legend(fontsize=8, loc="lower right", bbox_to_anchor=(0.3, 1.1), framealpha=0.7)


    # Line plot for interpolated electric potential
    axs[1].plot(df_input['datetime'], df_input['LastVoltageCh0'], label="CH0", color="blue")
    axs[1].plot(df_input['datetime'], df_input['LastVoltageCh1'], label="CH1", color="green", linestyle="dashed")

    # Labels and Titles
    axs[1].tick_params(axis='y', labelsize=10)
    axs[1].set_ylabel("Electric Potential (mV)",fontsize=10)
    axs[1].set_title("Adjusted Min-Max Scaled Input for CNN",fontsize=10)
    axs[1].legend(fontsize=8)

    # Temperature
    axs[2].tick_params(axis='y', labelsize=10)
    axs[2].set_ylabel("Temperature (°C)", fontsize=10)
    axs[2].plot(df_temp.index, df_temp["avg_leaf_temp"], label="Average Leaf Temperature", alpha=0.7)
    axs[2].plot(df_temp.index, df_temp["avg_air_temp"], label="Average Air Temperature", alpha=0.7)
    axs[2].set_title("Temperature Data", fontsize=10)
    axs[2].legend(fontsize=8, loc="lower right")

    # Improve spacing to prevent label cutoff
    fig.tight_layout()

    # Save figure in PGF format with proper bounding box
    plt.savefig("minMaxOnlineClassificationAdjusted525Shifted.pgf", format="pgf", bbox_inches="tight", pad_inches=0.05)
    #plot_path = os.path.join(save_dir, f"{prefix}_classified_plot.png")
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
    parser.add_argument("--data_dir", required=True, help="Directory with raw files.")
    parser.add_argument("--prefix", required=True, help="C1")
    parser.add_argument("--from_date", required=True, help="Cutoff date (YYYY-MM-DD). Cut off data before that date.")
    parser.add_argument("--until_date", required=True, help="Cutoff date (YYYY-MM-DD). Cut off data after that date.")
    args = parser.parse_args()

    # Normalize and validate inputs
    prefix = args.prefix.upper()
    data_dir = args.data_dir.lower()
    from_date = validate_date(args.from_date)
    until_date = validate_date(args.until_date)

    # Print Input Parameters
    console.print(f"[bold green]Data Directory:[/bold green] {data_dir}")
    console.print(f"[bold yellow]From:[/bold yellow] {from_date}")
    console.print(f"[bold yellow]Until:[/bold yellow] {until_date}")

    # Process Classified Data
    classified_files = discover_files(data_dir, prefix)
    df_classified = load_and_combine_csv(classified_files)
    df_classified['datetime'] = pd.to_datetime(df_classified['Datetime'], format='%Y-%m-%d %H:%M:%S:%f', errors='coerce')
    df_classified = df_classified.sort_values(by="datetime").reset_index(drop=True)
    df_classified = cut_data(df_classified, from_date, until_date)
    # Extract classification probabilities
    df_classified['ClassificationCh0_1'] = df_classified['ClassificationCh0'].str.extract(r'\[(?:\d+\.\d+),\s*(\d+\.\d+)\]').astype(float)
    df_classified['ClassificationCh1_1'] = df_classified['ClassificationCh1'].str.extract(r'\[(?:\d+\.\d+),\s*(\d+\.\d+)\]').astype(float)

    window_size = 100 # 100 = 10min

        # Compute rolling averages and store them in the dataframe
    df_classified["ch0_smoothed"] = df_classified["ClassificationCh0_1"].rolling(window=window_size, min_periods=1).mean()
    df_classified["ch1_smoothed"] = df_classified["ClassificationCh1_1"].rolling(window=window_size, min_periods=1).mean()

    threshold = 0.98

    # Create the final classification heat column
    df_classified["final_classification_heat"] = ((df_classified["ch0_smoothed"] > threshold) & 
                                                (df_classified["ch1_smoothed"] > threshold)).astype(int)



    # Convert VoltagesCh0NotScaled and VoltagesCh1NotScaled from string to lists of floats
    df_classified['VoltagesCh0'] = df_classified['VoltagesCh0NotScaled'].apply(lambda x: np.array(eval(x)))
    df_classified['VoltagesCh1'] = df_classified['VoltagesCh1NotScaled'].apply(lambda x: np.array(eval(x)))

    # Z-Score
    faktor = 1000
    df_classified['VoltagesCh0'] = df_classified['VoltagesCh0'].apply(lambda arr: faktor * ((arr - np.mean(arr)) / np.std(arr)) if np.std(arr) != 0 else arr)
    df_classified['VoltagesCh1'] = df_classified['VoltagesCh1'].apply(lambda arr: faktor * ((arr - np.mean(arr)) / np.std(arr)) if np.std(arr) != 0 else arr)


    # Extract the last voltage value for both channels
    df_classified['LastVoltageCh0'] = df_classified['VoltagesCh0'].apply(lambda x: x[-1])
    df_classified['LastVoltageCh1'] = df_classified['VoltagesCh1'].apply(lambda x: x[-1])

    df_input_nn = pd.DataFrame({
        "datetime": df_classified["datetime"],
        "LastVoltageCh0": df_classified["LastVoltageCh0"],
        "LastVoltageCh1": df_classified["LastVoltageCh1"]
    })
    num_rows = len(df_input_nn)
    print("Number of data points:", num_rows)
    # df_input_nn = df_input_nn.set_index("datetime", drop=False)
    # df_input_nn = df_input_nn.resample("1s").mean().interpolate()
    num_rows = len(df_input_nn)
    print("Number of data points:", num_rows)
    # min_max_scale_column(df_input_nn, "LastVoltageCh0")
    # min_max_scale_column(df_input_nn, "LastVoltageCh1")
    df_input_nn["LastVoltageCh0"] = df_input_nn["LastVoltageCh0"].rolling(window=window_size, min_periods=1).mean()
    df_input_nn["LastVoltageCh1"] = df_input_nn["LastVoltageCh1"].rolling(window=window_size, min_periods=1).mean()
    # df_input_nn = df_input_nn.set_index("datetime", drop=False)
    # df_input_nn = df_input_nn.resample("1s").mean().interpolate()

    # df_input_nn["time_diff"] = df_input_nn["datetime"].diff()
    # print(df_input_nn["time_diff"].value_counts())


    print(df_input_nn.head())


 
    # Process Temperature Node
    temp_files = discover_files(data_dir, "P6")
    df_temp = load_and_combine_csv(temp_files)
    df_temp = df_temp.rename(columns={'timestamp': 'datetime'})
    df_temp['datetime'] = pd.to_datetime(df_temp['datetime'], format='%Y-%m-%d_%H:%M:%S:%f', errors='coerce')
    df_temp = cut_data(df_temp, from_date, until_date)
    df_temp = resample_data(df_temp)
    df_temp["avg_leaf_temp"] = (df_temp["T1_leaf"] + df_temp["T2_leaf"]) / 2
    df_temp["avg_air_temp"] = (df_temp["T1_air"] + df_temp["T2_air"]) / 2

    # Calculate Arruracy
    preprocessed_dir = os.path.join(data_dir, "preprocessed")
    annotated_file = os.path.join(preprocessed_dir, "temp_annotated.csv")
    df_annotated = pd.read_csv(annotated_file)
    df_annotated["datetime"] = pd.to_datetime(df_annotated["datetime"], errors="coerce")
    df_annotated = df_annotated.dropna(subset=["datetime"])

    df_classified["datetime"] = df_classified["datetime"].dt.floor("s")
    df_annotated["datetime"] = df_annotated["datetime"].dt.floor("s")

    # Merge both DataFrames on "datetime" to ensure alignment
    df_merged = df_classified.merge(df_annotated, on="datetime", how="inner")

    # Save and Plot
    os.makedirs(preprocessed_dir, exist_ok=True)
    save_config_to_txt(CONFIG, preprocessed_dir, prefix)
    plot_data(df_classified, df_input_nn, df_merged, df_temp, prefix, threshold, preprocessed_dir)





    # Define correct classification cases
    correct_cases = (
        ((df_merged["final_classification_heat"] == 1) & df_merged["phase"].isin(["Increasing", "Holding"])) |
        ((df_merged["final_classification_heat"] == 0) & df_merged["phase"].isin(["Decreasing", "Nothing"]))
    )

    # Count correct cases
    correct_count = correct_cases.sum()

    # Total count (based on merged data)
    total_count = len(df_merged)

    # Compute accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0

    # Print the result
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")

    print("Classified Data Range:", df_classified["datetime"].min(), "to", df_classified["datetime"].max())
    print("Annotated Data Range:", df_annotated["datetime"].min(), "to", df_annotated["datetime"].max())

    # Check how many matching timestamps exist
    matching_datetimes = df_classified["datetime"].isin(df_annotated["datetime"])
    print(f"Matching timestamps: {matching_datetimes.sum()} / {len(df_classified)}")



if __name__ == "__main__":
    main()
