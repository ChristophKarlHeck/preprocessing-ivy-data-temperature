import argparse
import ast
import os
import re
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
    for i in range(idx + 1, min(idx + 201, len(df))):
        additional_values.append(df.loc[i, channel][-1])
    return current_list + additional_values  # 100 + 200 = 300 values

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
    if len(df) < 201:
        raise ValueError("Not enough rows in the CSV to create extended windows.")

    # Create the initial extended window for both channels (300 values each) from row 0.
    current_window_ch0 = get_extended_list(df, 0, 'input_not_normalized_ch0')
    current_window_ch1 = get_extended_list(df, 0, 'input_not_normalized_ch1')

    # Initialize the ground truth window with the next 200 rows' ground_truth values (rows 1 to 200)
    current_gt_window = [df.loc[i, "ground_truth"] for i in range(1, 201)]
    ground_truth = 0 if all(v == 0 for v in current_gt_window) else 1

    datetime = df.loc[200,"datetime"]

    result = []

    row_0 = {
        "datetime": datetime,
        "ground_truth": ground_truth,
        "input_not_normalized_ch0": list(current_window_ch0),
        "input_not_normalized_ch1": list(current_window_ch1),
    }

    result.append(row_0)

    # For demonstration, simulate sliding from row index 201 until the end.
    # At each sliding step, record the current window's last value (for both channels) and the aggregated GT.
    for i in range(201, len(df)):
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
    file_name = os.path.basename(args.file)

    print(file_name)
    match = re.search(r'_(C[12])', file_name)
    channel = None
    if match:
        channel = match.group(1)
        print("Extracted channel:", channel)
    else:
        print("Channel not found in the filename.")

    output_path = os.path.join(input_dir, f"input_not_normalized_{channel}_30min.csv")
    result_df.to_csv(output_path, index=False)

    


if __name__ == "__main__":
    main()