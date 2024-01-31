python3 ./src/data/sql_to_csv.py -i ./data/raw/zipcodes.germany.sql -o ./data/processed
pytest tests/test_data_processing.py
pytest tests/test_processed_data.py 