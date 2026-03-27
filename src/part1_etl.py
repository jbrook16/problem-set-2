'''
PART 1: ETL the two datasets and save each in a folder called `data/` as .csv's
'''

import pandas as pd
import os

def run_etl():
    """Load, process, and save the two datasets to CSV files."""
    os.makedirs('data', exist_ok=True)

    # Load the raw datasets
    pred_universe_raw = pd.read_csv('https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1')
    arrest_events_raw = pd.read_csv('https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1')

    # Convert filing_date to datetime
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw['filing_date'])
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw['filing_date'])

    # Drop the original filing_date column
    pred_universe_raw.drop(columns=['filing_date'], inplace=True)
    arrest_events_raw.drop(columns=['filing_date'], inplace=True)

    # Print dataset information
    print("=== pred_universe_raw ===")
    print(f"Shape : {pred_universe_raw.shape}")
    print(f"Cols  : {list(pred_universe_raw.columns)}")
    print(pred_universe_raw.head(3))
    print()
    
    print("=== arrest_events_raw ===")
    print(f"Shape : {arrest_events_raw.shape}")
    print(f"Cols  : {list(arrest_events_raw.columns)}")
    print(arrest_events_raw.head(3))
    print()

    # Save to CSV files
    pred_universe_raw.to_csv('data/pred_universe_raw.csv', index=False)
    arrest_events_raw.to_csv('data/arrest_events_raw.csv', index=False)
    
    print("✓ Saved  data/pred_universe_raw.csv")
    print("✓ Saved  data/arrest_events_raw.csv")

if __name__ == "__main__":
    run_etl()