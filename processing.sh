python3 ./src/data/sql_to_csv.py -i ./data/raw/zipcodes.germany.sql -o ./data/processed
python3 -m cudf.pandas src/test_processing.py