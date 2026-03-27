'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd

# Your code here
def run_preprocessing():

    # Load data
    pred_universe = pd.read_csv('data/pred_universe_raw.csv')
    arrest_events = pd.read_csv('data/arrest_events_raw.csv')

    # Merge (FULL OUTER JOIN)
    df_arrests = pd.merge(
        pred_universe,
        arrest_events,
        on='person_id',
        how='outer',
        suffixes=('_univ', '_event')
    )

    # Ensure datetime
    df_arrests['arrest_date_univ'] = pd.to_datetime(df_arrests['arrest_date_univ'])
    df_arrests['arrest_date_event'] = pd.to_datetime(df_arrests['arrest_date_event'])

    # -------------------------------
    # 🎯 TARGET VARIABLE (y)
    # -------------------------------
    df_arrests['y'] = 0

    for i, row in df_arrests.iterrows():
        if pd.isna(row['arrest_date_univ']):
            continue

        person = row['person_id']
        current_date = row['arrest_date_univ']

        # future window
        future = df_arrests[
            (df_arrests['person_id'] == person) &
            (df_arrests['arrest_date_event'] > current_date) &
            (df_arrests['arrest_date_event'] <= current_date + pd.Timedelta(days=365)) &
            (df_arrests['charge_degree'] == 'felony')
        ]

        if len(future) > 0:
            df_arrests.at[i, 'y'] = 1

    print("Share rearrested for felony in next year:", df_arrests['y'].mean())

    # -------------------------------
    # 🔹 FEATURE 1: Current felony
    # -------------------------------
    df_arrests['current_charge_felony'] = (
        df_arrests['charge_degree'] == 'felony'
    ).astype(int)

    print("Share of current charges that are felonies:",
          df_arrests['current_charge_felony'].mean())

    # -------------------------------
    # 🔹 FEATURE 2: Past felony count
    # -------------------------------
    df_arrests['num_fel_arrests_last_year'] = 0

    for i, row in df_arrests.iterrows():
        if pd.isna(row['arrest_date_univ']):
            continue

        person = row['person_id']
        current_date = row['arrest_date_univ']

        past = df_arrests[
            (df_arrests['person_id'] == person) &
            (df_arrests['arrest_date_event'] < current_date) &
            (df_arrests['arrest_date_event'] >= current_date - pd.Timedelta(days=365)) &
            (df_arrests['charge_degree'] == 'felony')
        ]

        df_arrests.at[i, 'num_fel_arrests_last_year'] = len(past)

    print("Average number of felony arrests last year:",
          df_arrests['num_fel_arrests_last_year'].mean())

    # Debug prints
    print("\nMean of num_fel_arrests_last_year:", df_arrests['num_fel_arrests_last_year'].mean())
    print("\ndf_arrests.head():")
    print(df_arrests.head())

    # Return for modeling
    return df_arrests


