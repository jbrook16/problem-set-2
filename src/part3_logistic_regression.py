'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr


def run_logistic_regression(df_arrests):
    """
    Run logistic regression with GridSearchCV to find optimal C hyperparameter.
    
    Args:
        df_arrests: DataFrame from preprocessing with features and target
        
    Returns:
        tuple: (df_arrests_test, gs_cv, X_test) for use in later parts
    """
    
    # Train-test split
    df_arrests_train, df_arrests_test = train_test_split(
        df_arrests,
        test_size=0.3,
        shuffle=True,
        stratify=df_arrests['y'],
        random_state=42
    )
    
    print(f"\n=== PART 3: Logistic Regression ===")
    print(f"Training set size: {len(df_arrests_train)}")
    print(f"Test set size: {len(df_arrests_test)}")
    
    
    features = ['current_charge_felony', 'num_fel_arrests_last_year']
    
    
    X_train = df_arrests_train[features]
    y_train = df_arrests_train['y']
    X_test = df_arrests_test[features]
    y_test = df_arrests_test['y']
    
    
    param_grid = {
        'C': [0.01, 0.1, 1.0]  # 0.01 = most regularization, 1.0 = least regularization
    }
    
    # Initialize model
    lr_model = lr(solver='lbfgs', max_iter=1000, random_state=42)
    
    
    gs_cv = GridSearchCV(
        lr_model,
        param_grid,
        cv=KFold_strat(n_splits=5),
        scoring='roc_auc'
    )
    
    
    gs_cv.fit(X_train, y_train)
    
    
    optimal_c = gs_cv.best_params_['C']
    print(f"\nOptimal C value: {optimal_c}")
    
    
    if optimal_c == 0.01:
        regularization = "most (strongest regularization)"
    elif optimal_c == 1.0:
        regularization = "least (weakest regularization)"
    else:
        regularization = "middle (moderate regularization)"
    
    print(f"This C value has: {regularization}")
    print(f"Best cross-validation score: {gs_cv.best_score_:.4f}")
    
    # Make predictions
    df_arrests_test = df_arrests_test.copy()
    df_arrests_test['pred_lr'] = gs_cv.predict_proba(X_test)[:, 1]
    
    print(f"\nPredictions added to test set")
    print(f"Test set shape: {df_arrests_test.shape}")
    
    return df_arrests_test, gs_cv, X_test


