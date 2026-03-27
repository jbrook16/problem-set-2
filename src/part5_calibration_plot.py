'''
PART 5: Calibration-light
- Read in data from `data/`
- Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Which model is more calibrated? Print this question and your answer. 

Extra Credit
- Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
- Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
- Compute AUC for the logistic regression model
- Compute AUC for the decision tree model
- Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10, title="Calibration Plot", filepath=None):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.
        title (str): Title for the plot.
        filepath (str): Path to save the plot. If None, displays plot.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(prob_true, bin_means, marker='o', label="Model", linewidth=2, markersize=8)
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath, dpi=100)
        print(f"  Saved plot to {filepath}")
        plt.close()
    else:
        plt.show()


def run_calibration_analysis():
    """
    Perform calibration analysis for LR and DT models.
    """
    
    print(f"\n PART 5: Calibration Analysis ")
    
    # Read prediction data from PART 4
    df_test = pd.read_csv('data/part4_test_predictions.csv')
    
    # Extract true labels and predictions
    y_true = df_test['y']
    pred_lr = df_test['pred_lr']
    pred_dt = df_test['pred_dt']
    
    print(f"\nTest set size: {len(df_test)}")
    print(f"Number of positives (y=1): {y_true.sum()}")
    print(f"Positive rate: {y_true.mean():.4f}")
    
    # Create calibration plots
    print("\n--- Logistic Regression Calibration Plot ---")
    calibration_plot(y_true, pred_lr, n_bins=5, title="Logistic Regression - Calibration Plot",
                     filepath='data/calibration_plot_lr.png')
    
    print("\n--- Decision Tree Calibration Plot ---")
    calibration_plot(y_true, pred_dt, n_bins=5, title="Decision Tree - Calibration Plot",
                     filepath='data/calibration_plot_dt.png')
    
    
    bin_means_lr, prob_true_lr = calibration_curve(y_true, pred_lr, n_bins=5)
    bin_means_dt, prob_true_dt = calibration_curve(y_true, pred_dt, n_bins=5)
    
    
    cal_error_lr = np.mean((prob_true_lr - bin_means_lr) ** 2)
    cal_error_dt = np.mean((prob_true_dt - bin_means_dt) ** 2)
    
    print(f"\n Calibration Assessment ")
    print(f"LR Calibration MSE: {cal_error_lr:.6f}")
    print(f"DT Calibration MSE: {cal_error_dt:.6f}")
    
    if cal_error_lr < cal_error_dt:
        better_calibrated = "Logistic Regression"
    else:
        better_calibrated = "Decision Tree"
    
    print(f"\nWhich model is more calibrated? {better_calibrated}")
    print(f"  (Lower calibration error = better calibrated)")
    
    
    print(f"\n EXTRA CREDIT ")
    
    # Top 50 predictions
    top_50_idx_lr = np.argsort(pred_lr)[-50:]
    top_50_idx_dt = np.argsort(pred_dt)[-50:]
    
    # PPV (Precision) for top 50
    y_pred_lr_top50 = np.zeros(len(pred_lr))
    y_pred_lr_top50[top_50_idx_lr] = 1
    ppv_lr = precision_score(y_true, y_pred_lr_top50)
    
    y_pred_dt_top50 = np.zeros(len(pred_dt))
    y_pred_dt_top50[top_50_idx_dt] = 1
    ppv_dt = precision_score(y_true, y_pred_dt_top50)
    
    print(f"\nPPV (Precision) for top 50 predictions:")
    print(f"  Logistic Regression: {ppv_lr:.4f}")
    print(f"  Decision Tree: {ppv_dt:.4f}")
    
    
    auc_lr = roc_auc_score(y_true, pred_lr)
    auc_dt = roc_auc_score(y_true, pred_dt)
    
    print(f"\nAUC Scores:")
    print(f"  Logistic Regression: {auc_lr:.4f}")
    print(f"  Decision Tree: {auc_dt:.4f}")
    
    
    ppv_winner = "Logistic Regression" if ppv_lr > ppv_dt else "Decision Tree"
    auc_winner = "Logistic Regression" if auc_lr > auc_dt else "Decision Tree"
    
    metrics_agree = ppv_winner == auc_winner
    
    print(f"\nDo both metrics agree that one model is more accurate?")
    print(f"  PPV winner: {ppv_winner}")
    print(f"  AUC winner: {auc_winner}")
    print(f"  Agreement: {'YES' if metrics_agree else 'NO'}")
    
    return df_test


if __name__ == "__main__":
    run_calibration_analysis()