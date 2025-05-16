import argparse
import ast
import os
import numpy as np
import pandas as pd

def slide_window(current_window, new_row_last_value):
    """
    Remove the first element from current_window and append new_row_last_value.
    Returns a window of the same length.
    """
    return current_window[1:] + [new_row_last_value]

def get_extended_list(df, idx, channel):
    """
    For row at index idx, build an extended list of 300 values:
      - Start with the current row's list (length = 100).
      - Append the last element from each of the next 200 rows.
    """
    current_list = df.loc[idx, channel]
    additional_values = []
    for i in range(idx + 1, min(idx + 501, len(df))):
        additional_values.append(df.loc[i, channel][-1])
    return current_list + additional_values  # 100 + 200 = 300 values


def downsample(lst: list, factor: int = 6) -> list[float]:
    """
    Downsample by averaging each non-overlapping block of `factor` values.
    E.g. factor=3 turns 300 → 100, factor=4 turns 400 → 100, etc.
    """
    arr = np.asarray(lst, dtype=float)
    n = arr.size
    if n % factor != 0:
        raise ValueError(f"Length {n} is not divisible by factor={factor}")
    down = arr.reshape(-1, factor).mean(axis=1)
    return down.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CSV file and simulate sliding windows with aggregated ground truth."
    )
    parser.add_argument("--file", required=True, help="CSV file containing the data.")
    args = parser.parse_args()

    print(f"Processing file: {args.file}")

    df = pd.read_csv(args.file)

    # Convert the string representations of lists into actual lists for channels.
    df['input_not_normalized_ch0'] = df['input_not_normalized_ch0'].apply(ast.literal_eval)
    df['input_not_normalized_ch1'] = df['input_not_normalized_ch1'].apply(ast.literal_eval)

    # ------------------------------
    # Build the initial extended windows:
    # ------------------------------
    # We need at least 200 future rows for a complete extended window.
    if len(df) < 501:
        raise ValueError("Not enough rows in the CSV to create extended windows.")

    # Create the initial extended window for both channels (300 values each) from row 0.
    current_window_ch0 = get_extended_list(df, 0, 'input_not_normalized_ch0')
    current_window_ch1 = get_extended_list(df, 0, 'input_not_normalized_ch1')

    # Initialize the ground truth window with the next 200 rows' ground_truth values (rows 1 to 200)
    current_gt_window = [df.loc[i, "ground_truth"] for i in range(1, 501)]
    ground_truth = 0 if all(v == 0 for v in current_gt_window) else 1

    datetime = df.loc[500,"datetime"]

    result = []

    row_0 = {
        "datetime": datetime,
        "ground_truth": ground_truth,
        "input_not_normalized_ch0": downsample(current_window_ch0),
        "input_not_normalized_ch1": downsample(current_window_ch0),
    }

    result.append(row_0)

    # For demonstration, simulate sliding from row index 201 until the end.
    # At each sliding step, record the current window's last value (for both channels) and the aggregated GT.
    for i in range(501, len(df)):
        # Record the sliding step (using an index starting at 0)

        # Slide the channel windows:
        new_val_ch0 = df.loc[i, 'input_not_normalized_ch0'][-1]
        new_val_ch1 = df.loc[i, 'input_not_normalized_ch1'][-1]
        current_window_ch0 = slide_window(current_window_ch0, new_val_ch0)
        current_window_ch1 = slide_window(current_window_ch1, new_val_ch1)
        datetime = df.loc[i,"datetime"]
        ground_truth = df.loc[i,"ground_truth"]

        row = {
            "datetime": datetime,
            "ground_truth": ground_truth,
            "input_not_normalized_ch0": list(current_window_ch0),
            "input_not_normalized_ch1": list(current_window_ch1),
        }
        result.append(row)

    result_df = pd.DataFrame(result)
    input_dir = os.path.dirname(args.file)
    # Split input_dir into its parent and the last folder name.
    parent_dir, last_folder = os.path.split(input_dir)
    # Append "_30min" to the last folder.
    new_folder = last_folder + "_60min"
    # Build the new output directory path.
    output_dir = os.path.join(parent_dir, new_folder)

    # Create the directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    # Build the output file path using the same file name.
    file_name = os.path.basename(args.file)
    output_path = os.path.join(output_dir, file_name)
    result_df.to_csv(output_path, index=False)
    print(f"Output written to {output_path}")

    


if __name__ == "__main__":
    main()