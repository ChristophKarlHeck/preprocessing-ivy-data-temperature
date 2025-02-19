#!/bin/bash

# Run preprocessing scripts with different parameters
python3 preprocessing.py --prefix P5 --data_dir /home/chris/experiment_data/1_2024_12_13-2024_12_19 --from_date 2024-12-14 --until_date 2024-12-19
python3 preprocessing.py --prefix P3 --data_dir /home/chris/experiment_data/1_2024_12_13-2024_12_19 --from_date 2024-12-14 --until_date 2024-12-19

python3 preprocessing.py --prefix P3 --data_dir /home/chris/experiment_data/2_2024_12_20-2024_12_26 --from_date 2024-12-20 --until_date 2024-12-26
python3 preprocessing.py --prefix P5 --data_dir /home/chris/experiment_data/2_2024_12_20-2024_12_26 --from_date 2024-12-20 --until_date 2024-12-26

python3 preprocessing.py --prefix P5 --data_dir /home/chris/experiment_data/3_2024_12_27-2025_01_03 --from_date 2024-12-27 --until_date 2025-01-03

python3 preprocessing.py --prefix P5 --data_dir /home/chris/experiment_data/4_2025_01_04-2025_01_08 --from_date 2025-01-04 --until_date 2025-01-08

python3 preprocessing.py --prefix P5 --data_dir /home/chris/experiment_data/5_2025_01_09-2025_01_15 --from_date 2025-01-10 --until_date 2025-01-16
python3 preprocessing.py --prefix P3 --data_dir /home/chris/experiment_data/5_2025_01_09-2025_01_15 --from_date 2025-01-10 --until_date 2025-01-16

python3 preprocessing.py --prefix P3 --data_dir /home/chris/experiment_data/6_2025_01_16-2025_01_23 --from_date 2025-01-17 --until_date 2025-01-20
python3 preprocessing.py --prefix P5 --data_dir /home/chris/experiment_data/6_2025_01_16-2025_01_23 --from_date 2025-01-17 --until_date 2025-01-20

python3 preprocessing.py --prefix P3 --data_dir /home/chris/experiment_data/7_2025_01_24-2025_01_30 --from_date 2025-01-25 --until_date 2025-01-30
python3 preprocessing.py --prefix P5 --data_dir /home/chris/experiment_data/7_2025_01_24-2025_01_30 --from_date 2025-01-25 --until_date 2025-01-30

python3 preprocessing.py --prefix P3 --data_dir /home/chris/experiment_data/8_2025_01_31-2025_02_11 --from_date 2025-02-01 --until_date 2025-02-10
python3 preprocessing.py --prefix P5 --data_dir /home/chris/experiment_data/8_2025_01_31-2025_02_11 --from_date 2025-02-01 --until_date 2025-02-10