# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:03:03 2022

@author: aossipov
"""

import numpy as np
import pandas as pd

filename = 'K:\Alexander Ossipov\Projects\Generic\Input_Data\performance_summary.csv'

filename_out = 'K:\Alexander Ossipov\Projects\Generic\Input_Data\performance_summary_reordered.csv'

df_performance = pd.read_csv(filename, index_col=0)

cols = ['label_columns','sharpe', 'sortino', 'features', 'model', 'model_parameters', 'validation',
       'train_first_len', 'val_len', 'train_len', 'test_len',
       'fixed_size_training', 'overlap_training', 'delta', 'num_days',
       'input_width', 'mean_PnL', 'total PnL', 'total volume', 'num_positions',
       'max_drawdown', 'drawdown_to_pnl', 'RMSE', 'R^2',
       'error_autocorr', 'decision_accuracy', 'decision_AUC',
       'position_size_col', 'first_date', 'last_date']

df_performance = df_performance[cols]


df_performance.to_csv(filename_out)
