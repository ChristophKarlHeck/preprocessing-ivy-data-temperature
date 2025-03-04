## 1. Do preprocessing for each week and phyto-node
```bash
python3 preprocessing.py --data_dir /home/chris/experiment_data/11_2025_02_27-2025_03_04 --from_date 2025-02-27 --until_date 2025-03-04
```

```bash
python3 annotating_only_temp.py --data_dir "/home/chris/experiment_data/11_2025_02_27-2025_03_04/preprocessed"
```

```bash
python3 check_classification.py --data_dir "/home/chris/experiment_data/11_2025_02_27-2025_03_04" --prefix C1 --from_date 2025-02-27 --until_date 2025-03-04
```