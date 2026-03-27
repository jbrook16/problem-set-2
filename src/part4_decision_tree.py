'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Save dataframe(s) save as .csv('s) in `data/`
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC


def run_decision_tree(df_arrests, df_arrests_test_lr, X_test_lr):
    """
    Run decision tree with GridSearchCV to find optimal max_depth hyperparameter.
    
    Args:
        df_arrests: DataFrame from preprocessing with features and target
        df_arrests_test_lr: Test set from PART 3 with LR predictions
        X_test_lr: Feature matrix for test set from PART 3
        
    Returns:
        df_arrests_test: Test set with both LR and DT predictions
    """
    
    # Train-test split 
    df_arrests_train, df_arrests_test = train_test_split(
        df_arrests,
        test_size=0.3,
        shuffle=True,
        stratify=df_arrests['y'],
        random_state=42
    )
    
    print(f"\n=== PART 4: Decision Tree ===")
    print(f"Training set size: {len(df_arrests_train)}")
    print(f"Test set size: {len(df_arrests_test)}")
    
    # Define features 
    features = ['current_charge_felony', 'num_fel_arrests_last_year']
    
    # Prepare data
    X_train = df_arrests_train[features]
    y_train = df_arrests_train['y']
    X_test = df_arrests_test[features]
    y_test = df_arrests_test['y']
    
    
    
    param_grid_dt = {
        'max_depth': [3, 5, 10]  # 3 = most regularization, 10 = least regularization
    }
    
    
    dt_model = DTC(random_state=42)
    
    
    gs_cv_dt = GridSearchCV(
        dt_model,
        param_grid_dt,
        cv=KFold_strat(n_splits=5),
        scoring='roc_auc'
    )
    
    
    gs_cv_dt.fit(X_train, y_train)
    
    # max_depth
    optimal_depth = gs_cv_dt.best_params_['max_depth']
    print(f"\nOptimal max_depth value: {optimal_depth}")
    
    
    if optimal_depth == 3:
        regularization = "most (strongest regularization, shallowest tree)"
    elif optimal_depth == 10:
        regularization = "least (weakest regularization, deepest tree)"
    else:
        regularization = "middle (moderate regularization)"
    
    print(f"This max_depth has: {regularization}")
    print(f"Best cross-validation score: {gs_cv_dt.best_score_:.4f}")
    
    
    df_arrests_test_combined = df_arrests_test_lr.copy()
    df_arrests_test_combined['pred_dt'] = gs_cv_dt.predict_proba(X_test_lr)[:, 1]
    
    print(f"\nPredictions added to test set")
    print(f"Test set shape: {df_arrests_test_combined.shape}")
    
    # Save to CSV
    df_arrests_test_combined.to_csv('data/part4_test_predictions.csv', index=False)
    print(f"\n✓ Saved data/part4_test_predictions.csv")
    
    return df_arrests_test_combined, gs_cv_dt