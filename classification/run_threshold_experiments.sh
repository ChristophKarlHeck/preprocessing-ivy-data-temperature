#!/bin/bash

# Force numeric formatting with a period as the decimal separator
export LC_NUMERIC=C

# Loop over thresholds from 0.0 to 1.0 in steps of 0.01
for threshold in $(seq 0 0.01 1); do
    # Format the threshold to two decimal places (ensuring a period is used)
    threshold_formatted=$(printf "%.2f" "$threshold")
    echo "Running with threshold: $threshold_formatted"
    python3 check_classification.py \
        --data_dir "/home/chris/experiment_data/10_2025_02_20-2025_02_27" \
        --prefix C1 \
        --from_date "2025-02-24 07:00" \
        --until_date "2025-02-24 19:30" \
        --threshold "$threshold_formatted" \
        --plant_id 532
done