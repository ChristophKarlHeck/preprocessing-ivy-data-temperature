## 1. Do preprocessing for each week and phyto-node
```bash
python3 preprocessing.py --data_dir /home/chris/experiment_data/11_2025_02_27-2025_03_04 --from_date 2025-02-27 --until_date 2025-03-04
```

```bash
python3 annotating_only_temp.py --data_dir "/home/chris/experiment_data/11_2025_02_27-2025_03_04/preprocessed"
```

```bash
python3 check_classification.py --data_dir /home/chris/experiment_data/13_2025_03_11-2025_03_14 --prefix C1 --from_date "2025-03-11 10:00" --until_date "2025-03-15 16:00" --threshold 0.7 --plant_id 3
```

```bash
python3 check_classification.py --data_dir /home/chris/experiment_data/15_2025_04_08-2025_04_14 --prefix C1 --from_date "2025-04-11 06:00" --until_date "2025-04-13 18:30" --threshold 0.5 --plant_id 3
```


```bash
python plot_precision.py --data-dir "/home/chris/experiment_data/10_2025_02_20-2025_02_27" --file "threshold_engineering_C1.csv"
```