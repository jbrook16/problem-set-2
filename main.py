'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
from src import etl
from src import preprocessing
from src import logistic_regression
from src import decision_tree
from src import calibration_plot



# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    etl.run_etl()

    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests = preprocessing.run_preprocessing()

    # PART 3: Call functions/instanciate objects from logistic_regression
    df_arrests_test_lr, lr_model, X_test_lr = logistic_regression.run_logistic_regression(df_arrests)

    # PART 4: Call functions/instanciate objects from decision_tree
    df_arrests_test_combined, dt_model = decision_tree.run_decision_tree(df_arrests, df_arrests_test_lr, X_test_lr)

    # PART 5: Call functions/instanciate objects from calibration_plot
    df_results = calibration_plot.run_calibration_analysis()


if __name__ == "__main__":
    main()