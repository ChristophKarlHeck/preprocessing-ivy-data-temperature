# preprocessing-ivy-data-temperature

## 1. Do preprocessing for each week and phyto-node
```bash
python3 preprocessing.py --prefix P5 --data_dir /home/chris/experiment_data/5_09.01.25-15.01.25 --from_date 2025-01-10 --until_date 2025-01-16
```

## 2. Phase annotation 
```bash
python3 annotating.py --data_dir "/home/chris/experiment_data/5_09.01.25-15.01.25/preprocessed"
```

## 3. Splitting to final document
```bash
python3 splitting.py --data_dir "/home/chris/experiment_data/5_09.01.25-15.01.25" --split_minutes 10 --prefix "P3"
```

## OR
```bash
python3 extract_important_domains.py --data_dir /home/chris/experiment_data/7_2025_01_24-2025_01_30 --prefix "P3" --before 30 --after 30 --split_minutes 10
```

## Utils
Only for exceptions