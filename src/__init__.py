# Import modules with convenient aliases
from . import part1_etl as etl
from . import part2_preprocessing as preprocessing
from . import part3_logistic_regression as logistic_regression
from . import part4_decision_tree as decision_tree
from . import part5_calibration_plot as calibration_plot

__all__ = ['etl', 'preprocessing', 'logistic_regression', 'decision_tree', 'calibration_plot']
