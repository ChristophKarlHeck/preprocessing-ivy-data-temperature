# preprocessing-ivy-data-temperature

## Do preprocessing for each week and phyto-node
```bash
python3 preprocessing.py --prefix P5 --data_dir /home/chris/experiment_data/5_09.01.25-15.01.25 --from_date 2025-01-10 --until_date 2025-01-16
```

## Phase annotation 
```bash
python3 annotating.py --data_dir "/home/chris/experiment_data/5_09.01.25-15.01.25/preprocessed"
```
## Utils
Only for exceptions