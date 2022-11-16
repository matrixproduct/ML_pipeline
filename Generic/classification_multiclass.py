# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:05:14 2021

@author: aossipov

Generic script for classification, in which new methods are implemented first.
Can be used as a template for particular projects.

"""

import numpy as np
import pandas as pd
import copy




from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from itertools import product

import matplotlib.pyplot as plt

import sys, importlib, os
    

sys.path.append('C:/tsLocal/folib/ML_tools')

import train_validation_test as tvt

from  train_validation_test import  plot_profit, \
                                    get_label_columns, \
                                    stability_cross_validation,\
                                    prob_to_label,\
                                    save_model_results, not_all_equal,\
                                    TrainTest, randomised_cross_validation_performance,\
                                    binary_return , num_days_to_str,\
                                    compare_models, exclude_intervals,\
                                    DeterministicClassifier,\
                                    create_error_model


                                    
                                
sys.path.append('C:/tsLocal/users/alexander_ossipov/Projects/COT/')

from cot_data_load import get_data


import features

importlib.reload(features)

import shap

from features import clustered_MDA_feature_imp, mutual_stat_significance
# from features import greedy_feature_selection, rolling_quantile, \
#                     conditional_correlation_stat, clustered_MDA_feature_imp
                     



from sklearn.tree import plot_tree



sys.path.append('C:/tsLocal/users/alexander_ossipov/Projects/Generic')
from price_data_load import get_symbol_data




path_prefix = '//trs-fil01/trailstone/Alexander Ossipov/Projects/Generic/'
input_path_prefix = '//trs-fil01/trailstone/Alexander Ossipov/Projects/Generic/Input_Data'
output_path_prefix = '//trs-fil01/trailstone/Alexander Ossipov/Projects/Generic/Output_Data'
plot_path_prefix = '//trs-fil01/trailstone/Alexander Ossipov/Projects/Plots/'
# output_path_prefix = 'C:/temp'



# %% Parser for command-line arguments

def ParseArgs():
    ''' Parse Arguments '''
    import argparse
    parser = argparse.ArgumentParser(description='description',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
    parser.add_argument('--num_days', type=int, 
                        help='number of trading days or time shift for future prediction', 
                        default=None)
    parser.add_argument('--input_width', type=int, 
                        help='number of rows used to create features from a given column of df or time lag', 
                        default=None)
    
    parser.add_argument('--trading', action='store_true', 
                        help=' if True, the model is for trading, i.e. PnL and other trading metrics can be calculated', 
                        default=None)
    
    parser.add_argument('--no-trading',  dest='trading', action='store_false',   
                        help=' if True, the model is for trading, i.e. PnL and other trading metrics can be calculated'
                        )
     
    parser.add_argument('--price_prediction', action='store_true', 
                        help='if True, the model predict price rather than return', 
                        default=None)
    
    parser.add_argument('--no-price_prediction', dest='price_prediction', action='store_false',  
                        help='if True, the model predict price rather than return' 
                        )
    args = parser.parse_args()
    
    return args

args = ParseArgs()

# %% Create additional features (and target) 



def create_features_target(df, sym1=None, sym2=None, num_days=None, **kwargs):
    
    num_days_str =  num_days_to_str(num_days)  
    day_ahead = kwargs.get('day_ahead', False)   
    if day_ahead:
        symbols = [sym1] if sym2 is None else [sym1, sym2]
        for sym in symbols:
            if f'{sym}_da' in df.columns:
                df[f'{sym}_da_spread' ] = df[f'{sym}_da'] - df[sym]
                df[f'{sym}_da_spread return{num_days_str}'] = df[f'{sym}_da return{num_days_str}']\
                                                        -  df[f'{sym} return{num_days_str}']
                df[f'{sym}_da_spread binary return{num_days_str}'] = binary_return(df[f'{sym}_da_spread return{num_days_str}'])                                    
      
   
    
    df.dropna(inplace=True)
    return df


##############################################################################
##############################################################################

if __name__ == '__main__':

# %% General parameters
    
    # dates for data loading
    # if filename is provided in get_symbol_data() and the file exists,
    # data will be loaded from a file with respective dates
    # eff_start, eff_end = '2000-01-01', '2022-01-19' 
    eff_start, eff_end =  '2015-01-01', '2022-03-13'    
   
    # symbols = {name to be used in DataFrame: tsArctic_name}
    # symbols = {'ttf': 'ice.code.tfm', 'gpw': 'ice.code.gab', 'coal': 'ice.code.atw', 'wti':'cme.code.cl'}
    symbols = {'ttf': 'ice.code.tfm'}
    
    # number of trading dates, set it to 0, if model is not for trading
    # the target will be generated by shifting label_columns by num_days
    num_days = 1
    if args.num_days:
       num_days = args.num_days
    
    
    cal = 'nb01'  # calendar used in tsArctic
    cal2 = None   # the second one is optional, if not requires set to None
    # cal2 = 'nb02'
    cal_str = f'_cal={cal}'  # string for calendar in file name
    cal2_str = f'_cal2={cal2}' if cal2 is not None else ''
    cal_str += cal2_str
    
    # sym1 (and optionally sym2) symbols names used to select the target below
    # they must be from symbols.keys()
    sym1 = 'ttf'
    # sym1 = 'nbp'
    # sym1 = 'gpw'
    # sym2 = 'ttf'
    # sym2 = 'gpw'
    sym2 = None 
    
    # sym_name is constructed from from sym1 (and optianlly sym2)
    # sym_name =  f'{sym1}_{sym2}'  # spread between sym1 and sym2
    sym_name = f'{sym1}'  # price of sym1
    # sym_name =  f'{sym1}_spread'  # spread between cal and cal2 for sym1
    # sym_name =  f'{sym1}_da_spread'  # spread between cal and day ahead
    
    day_ahead=True  # if True, day ahead price data will be loaded
    # day_ahead=False
    
   
    # used as a part of filename related to symbols
    symbols_names = ''
    for symbol in symbols:
        symbols_names += f'_{symbol}'
        
    # symbols_filename =  f'{input_path_prefix}/price_data{symbols_names}{cal_str}_num_days={num_days}.csv'
    symbols_filename = f'{input_path_prefix}/price_data_ttf_da.csv'
    
   # data for prices, spreads etc
   # if file with filename exists, data is loaded from file
   # if filename is provided, but file doesn't exist, data is load from Arctic and saved to the file
   # if filename is not provided, data is load from Arctic
    df = get_symbol_data(symbols, eff_start, eff_end, num_days, 
                        cal=cal, cal2=cal2, drop_nan=True, day_ahead=True,
                        filename=symbols_filename)
  
    
    # data for ttt day ahead spread
    
    # all_data_filename = f'{input_path_prefix}/price_data_ttf_da.csv'
    
    # more general function getting data not only for prices,
    # but also for other project specific features. 
    #It should be modified for each particular project
    # df = get_data(symbols, eff_start, eff_end, num_days, 
    #           cal=cal, cal2=cal2, drop_nan=True, 
    #           countries=countries,
    #           symbols_filename=symbols_filename, 
    #           ldz_forecast_filename=ldz_forecast_filename, 
    #           filename=all_data_filename,
    #           var_filename=var_filename,
    #           day_ahead=day_ahead)
    
   
    
    
    
    file_suffix = f'{sym_name}_eff_end={eff_end}_trade_num_days_{num_days}_cal={cal}'
    file_suffix_model = f'{sym_name}_trade_num_{num_days}_cal={cal}'
    
    
    delta=0.1 # threshold for converting probability prediction to label
    
    n_splits = 10  # number of splits in cross-validation
    
    tti = None  # train test instance 
    
    df['random_target'] = np.random.choice(3, len(df))
    
# %% Train/test length

    train_first_len = 1  # length of the first train set, not currently used, keep =1 
    val_len = 1  # length of validation length, not currently used, keep =1
     
    
    # train_len = 300
    # test_len = 75j
    
    train_frac = 0.8
    val_frac = 0.
    
    # train_len = int(len(df) * train_frac)
    train_len = 375
    val_len = int(len(df) * val_frac)
    # test_len = int(len(df_bbg) * (1 - train_frac - val_frac)) - 3
    test_len = 25
    
    cross_val_len = int(len(df) * 0.8) 
    
    # number of rows used to create features from a given column of df
    # df.loc[t, column], df.loc[t - 1, column], ..., df.loc[t - (input_width - 1), column]
    # will be used as different features 
    input_width = 1 
    
    if args.input_width:
        input_width = args.input_width
   
# %% Target       
    
    # if True the model is for trading, i.e. PnL and other trading metrics can be calculated
    # trading = True  
    trading = False
    if args.trading:
        trading = args.trading
    
    
    # trading models may predict returns or price
    # price_prediction = True  # if true we predict price rather than return
    price_prediction = False
    if args.price_prediction:
        price_prediction = args.price_prediction
    # used to specify the target, usually the same as sym_name
    target_sym = sym_name
     
    # if trading:
    #     if not price_prediction:
    #         # generate continuous and binary label columns
    #         label_columns =  get_label_columns(target_sym, 1)  # generate continuous and binary label columns
    #         # label_columns =  get_label_columns(target_sym, num_days)
    #     else:    
    #         # for regression the binary columns plays no role
    #         # so we choose an abitrary binary column as the second label column
    #         binary_cols = [col for col in df.columns if 'binary' in col]
    #         label_columns = [f'{sym_name}', binary_cols[0]]  
    # else:    
    #     binary_cols = [col for col in df.columns if 'binary' in col]
    #     label_columns = [f'{sym_name}', binary_cols[0]]  
    
    label_columns = [target_sym, 'random_target']
  
# %% Features    
    
    # create additional features
    df = create_features_target(df, sym1, sym2, num_days=1, day_ahead=day_ahead)
                              
    
    # exclude some dates from data, if necessary
    df = df[(df.index < '2018-02-23' ) | (df.index > '2018-06-03')]  # exclude some days with crazy target values

    # reduce data for a quicker check
    df = df[-500:]

    # select features
    selected_columns = ['ttf', 'ttf return', 'ttf_da',
          'ttf_da return', 'ttf binary return', 
          'ttf_da binary return',  
          'ttf_da_spread'
          ]   


    
# %% Models
    
    
    # lgbm_model = LGBMClassifier(max_depth = 10, n_estimators = 10, lambda_l1 = 0.5,  lambda_l2 = 0.5, random_state = 111)

    # tree_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=1, random_state=566)
    
    rf_model = RandomForestClassifier(max_depth=3, n_estimators=60, min_samples_leaf=0.05, random_state=566)

    logistic_reg_model = LogisticRegression()

    # clf = DecisionTreeClassifier(criterion='entropy',
    #                             max_features=1,  
    #                             min_weight_fraction_leaf=0, random_state=566)

    # clf = BaggingClassifier(base_estimator=clf, 
    #                   n_estimators=2000, 
    #                   max_features=1., 
    #                   max_samples=1., 
    #                   oob_score=False, random_state=111)


    # model = lgbm_model
    # model = lgbm_model_new
    # model = tree_model
    model = rf_model
    # model = clf
    # model = logistic_reg_model

        
    
    
   
# %% Sample weights
# Set sample weights for training

    sample_weight_col = None
    # sample_weight_col = 'sample_weight'
    
    if sample_weight_col is not None:
    
        df['sample_weight'] = abs(df[label_columns[0]].shift(-num_days))  # weight = abs(price_change)

# %% Exclude dates from training
    exclude_index = None
    # exclude_index = ['2017-12-15', '2017-12-18', '2017-12-19']
    
# %% Exclude values of feature
    feature = 'ttf'
    intervals = [(0, 1), (2.4, 3.1)]
    # df = exclude_intervals(df, feature, intervals 
# %% Control flow

    # get_mean_target_return = True
    get_mean_target_return = False
    
    
    position_size_col = None
    # position_size_col = f'{sym_name} Position Size'
    # position_from_prediction = True
    position_from_prediction = False
      

    # greedy_search = True
    greedy_search = False

    # bin_stat_single_target = True
    bin_stat_single_target = False
    
    # bin_stat = True
    bin_stat = False
    
    # explore_stability_cross_val = True
    explore_stability_cross_val = False
    
    # explore_stability_test = True
    explore_stability_test = False
    
    # run_cross_validation = True
    run_cross_validation = False
    
    run_test_object=True
    # run_test_object=False
    
    # bin_metrics = True
    bin_metrics = False
    
    # performance_groups = True
    performance_groups = False
    
    # plot_tree_decisions = True
    plot_tree_decisions = False
    
    # random_noise = True
    random_noise = False

    # clustering = True
    clustering = False
     
    # sulov = True
    sulov = False
    
    # make_prediction = True
    make_prediction = False
    
    # explore_shap = True
    explore_shap = False
    
    # save_deployment_model = True 
    save_deployment_model = False
    
    # make_total_prediction = True
    make_total_prediction = False
    
    # model_for_errors = True
    model_for_errors = False  
    
    model_for_error_as_feature = True
    # model_for_error_as_feature = False
    
    # calculate_confusion_matrix = True
    calculate_confusion_matrix = False
    
# # %% Feature selection by  clustering
# currently not completed   
#     exclude_columns = []
    
#     df_selected = df[selected_columns]
    
    
   
    
    
    
    
   
    
#     total_val_len = train_len
#     val_train_factor = 0.8
#     val_test_factor = 0.3
#     val_train_len = int(val_train_factor * total_val_len)
#     val_test_len =  int(val_test_factor*(total_val_len - val_train_len)) 
#     maxNumClusters=10
#     n_init=10
#     n_splits=10
#     num_real=400
#     random_state=31
#     cluster_selection=True
#     select_metric='sortino'
#     verbose=0 
    
    
    
#     clusters, importance, best_clusters, exclude_columns, all_removed_features = feature_clustering_selection(df, target_sym, model, total_val_len, 
#                                     val_train_len, val_test_len, 
#                                     exclude_columns, delta, num_days, 
#                                     maxNumClusters, n_init, n_splits, num_real, random_state,
#                                     feature_selection=cluster_selection, select_metric=select_metric,
#                                     verbose=verbose)
    
   
#     ##########################################################################
#     # Second clustering
#     ##########################################################################
#     all_features = list(df.columns)
#     best_clusters_reduced = {}
#     print('\n*******  Second clustering *******\n')
#     print('Loop over clusters')
#     for cluster_ind, cluster in best_clusters.items():
#         print('\nCluster', cluster_ind )
#         exclude_columns = [f for f in all_features if f not in cluster] 
#         new_clusters, new_importance, new_best_clusters, new_exclude_columns, new_all_removed_features = feature_clustering_selection(df, target_sym, model, total_val_len, 
#                                     val_train_len, val_test_len, 
#                                     exclude_columns, delta, num_days, 
#                                     maxNumClusters, n_init, n_splits, num_real, random_state,
#                                     feature_selection=cluster_selection, select_metric=select_metric,
#                                     verbose=verbose)
        
#         # if not all_removed_cols:
#         reduced_cluster = [f for f in cluster if f not in new_all_removed_features]
#         best_clusters_reduced[cluster_ind] = reduced_cluster
#         all_removed_features.append(new_all_removed_features)
    
#     print('Best clusters reduced:', best_clusters_reduced)
#     print('All removed features:', all_removed_features)


# %% Mean taregt return
# Calculate mean, median and std values for target


    if get_mean_target_return:
    
        if tti is None:
            
               
            # validation = True
            validation = False
            
            fixed_size_training = True 
            overlap_training = True 
            
        
            tti = TrainTest(df, model, validation, train_first_len, val_len, train_len,\
                                                test_len, label_columns, selected_columns=selected_columns,\
                                                fixed_size_training=fixed_size_training, overlap_training=overlap_training,
                                                input_width=1, label_width=1, num_days=num_days, scaling=False)
            
        mean_tr, median_tr, std_tr = tti.mean_target_return() 
    
        print(f'\nMean abs target return: {mean_tr: .2f}, median abs: {median_tr: .2f}, standard deviation: {std_tr: .2f} \n')       
    


# %% Features selection by greedy search
        
   
    if greedy_search:
        
        ##########################################################################
        # Remove less important features by greedy search
        ##########################################################################
        print('\n*******  Remove less important features by greedy search *******:\n')
        
        
         
        
        # validation = True
        validation = False
        
        fixed_size_training = True 
        overlap_training = True 
        
        
        # init TrainTest innstance (tti) if it doesn't exist
        
        if tti is None:
        
            tti = TrainTest(df, model, validation, train_first_len, val_len, train_len,\
                                                test_len, label_columns, selected_columns=selected_columns,\
                                                fixed_size_training=fixed_size_training, overlap_training=overlap_training,
                                                input_width=1, label_width=1, num_days=num_days, scaling=False)
            
        
        
        feature_rank = False
        position_size_col = None
        n_features_penalty = 0.05
        
        # cols = df.columns
        cols = selected_columns
        # cols = selected_columns
        
        
        
        removed_features = tti.greedy_feature_selection(cols, n_splits,
                                                    delta=delta, reverse=True, 
                                                    select_metric='sortino', verbose=2, 
                                                    feature_rank=feature_rank,  
                                                    position_size_col=position_size_col, 
                                                    n_features_penalty=n_features_penalty)
       
     
        print('\n Removed features: ', removed_features)
        # exclude_columns = removed_features
       
# %%   Feature imporatnce using bin stats target 
    ##########################################################################
    # Feature imporatnce using bin stats
    ##########################################################################
    # Bin every feature from cols_to_bin in df  into bins
    # then calculate mean, std, and skew for target
    # feature in df w.r.t. that bins.
  
   
    if bin_stat_single_target:
        print('\n*******  Feature imporatnce using bin stats for target *******:\n')

      
        output_bin_stat_file = f'{output_path_prefix}/bin_stat_{file_suffix}.csv'
    
        # label_columns =  get_label_columns(target_sym, num_days)
        
        X_train = df[:train_len]
        # X_train = df_quarter
        y = X_train[label_columns[0]].shift(-num_days)[:-num_days]
        X = X_train[:-num_days] 
        
        
        df_with_target = X.copy()
        df_with_target['target'] = y
    
        
        cols_to_bin = df.columns
        # cols_to_bin = selected_columns
        bins = 10
        
        df_stat_dict, target_stat, target_stat_binary = features.get_bin_stat_target(df_with_target, cols_to_bin, 'target', bins, quantiles=True, best_to_plot = 0)
        
        target_stat_copy = target_stat.copy()
        target_stat_copy = target_stat_copy[target_stat_copy['switching']<4]
        target_stat_copy.sort_values(by=['max_min'], ascending=False, inplace=True)       
        selected_feat = target_stat_copy.index[:10]
        for feat in selected_feat:
            features.plot_bin_one_target(df_stat_dict[feat])
# %% Feature imporatnce using bin stats
    ##########################################################################
    # Feature imporatnce using bin stats
    ##########################################################################
    # Bin every feature from cols_to_bin in df  into bins
    # then calculate mean and std for all features in df w.r.t. that bins 
    # or their combinatorial products.
    # Generalisation of the above block "Feature imporatnce using bin stats target".
    # It can be useful for identifying a potential target.
   
    if bin_stat:
        print('\n*******  Feature imporatnce using bin stats *******:\n')

      
        output_bin_stat_file = f'{output_path_prefix}/bin_stat_{file_suffix}.csv'
    
        # label_columns =  get_label_columns(target_sym, num_days)
        
        X_train = df[:train_len]
        # X_train = df_quarter
        y = X_train[label_columns[1]].shift(-num_days)[:-num_days]
        X = X_train[:-num_days] 
        
        
        df_with_target = X.copy()
        df_with_target['target'] = y
        
        
        # cols_to_bin = [col for col in df_quarter.columns if 'target' not in col]
        # target_cols = [col for col in df_quarter.columns if 'target' in col]
        # bins = 10
        
        # # df_stat_list = get_bin_stat(df_quarter, cols_to_bin, bins)
        
        # # df_stat_list[0] 
        
        
        
        cols_to_bin = df.columns
        bins = 5
        # cols_to_bin = df_quarter.columns
        
        # cols_to_bin = ['WTI Managed Money Long F+ O change', 'WTI Managed Money Short F+ O change', 'WTI quantile',
        #                     'WTI Managed Money Long F+ O quantile', 'WTI Managed Money Short F+ O quantile']
        
    
        
        selected_cols = 'target'
        
        
        # df_single = mutual_stat_significance(df_quarter, cols_to_bin, bins, selected_cols, quantiles=True) 
        # df_single = mutual_stat_significance(df_quarter, cols_to_bin, bins, target_cols, quantiles=True) 
        
        df_single = mutual_stat_significance(df_with_target, cols_to_bin, bins, selected_cols, quantiles=True, verbose=True) 
        # get_bin_stat(df_quarter, 'TTF Q1', bins)
            
        target_col = 'target'
        df_single.sort_values(by=[target_col], ascending=False, inplace=True)
        
        print(df_single[:60])
        
        total_num_features = len(df_single)
        # num_single = total_num_features - 1
        num_single = 20
        best_single_features = list(df_single.index[:num_single].values)
        print(best_single_features)
        
        # save results
        df_single.to_csv(output_bin_stat_file)

        # a = df.columns
        # X_train = df[:train_len]
        # y = X_train[label_columns[0]].shift(-num_days)[:-num_days]
        # X = X_train[:-num_days]   
        # df_with_target = X.copy()
        # df_with_target['target'] = y
        # cols_pairs = list(product(a,a))
        # bins = 6
      
        
        # df_conditional_correlation, corr_info = conditional_correlation_stat(df_with_target, cols_pairs, bins, 'target', quantiles=True)    
    

    
# %%  Stability cross-validation
   
    if explore_stability_cross_val:
                
        
        ########################################
        # Stability cross-validation analysis  #
        ########################################
        # Test the satbility of predictions for cross-validation by perturbing
        # features and comparing how the predictions change.
        
        
        validation = False
        
        fixed_size_training = True 
        overlap_training = True 
        overlap_trade = False
        
        
        # change_mode = 'difference'
        change_mode = 'std'
        roll_window = 10
        num_points = 3
        epsilon = 0.1
        
        cols_to_change = selected_columns[:3]
        
        output_stability_file = f'{output_path_prefix}/stability_test_prob_predictions_{file_suffix}_eps={epsilon}_change_mode={change_mode}.csv'
       
        tti = TrainTest(df, model, validation, train_first_len, val_len, train_len,\
                                        test_len, label_columns, selected_columns=selected_columns,\
                                        fixed_size_training=fixed_size_training, overlap_training=overlap_training,
                                        input_width=1, label_width=1, num_days=num_days, scaling=False)
        
        
        tti.cross_validation(n_splits=n_splits, filename=None, verbose=2)
        
        
        predictions_perturbed = tti.stability_test(cols_to_change, num_points, \
                                epsilon, change_mode, roll_window, input_set='cross_val', filename=None)
             
            
           
        predictions_perturbed.loc[:,'Prediction'] = predictions_perturbed.loc[:, tti.pred_col_names].values.argmax(axis=1)     
        
        # stability_df = prob_predictions['Prediction'].groupby(level=[0,1]).apply(lambda a: is_middle_opposite(a.values))
        stability_df = predictions_perturbed['Prediction'].groupby(level=[0,1]).apply(lambda a: not_all_equal(a.values))
        
        
        stability_by_date = stability_df.groupby(level=0).sum()
        stability_by_feature = stability_df.groupby(level=1).sum()
        
        
        
        
        
# %%  Stability test
   
    if explore_stability_test:
                
        
        ############################
        # Stability test analysis  #
        ############################
        # Test the satbility of predictions for the test set by perturbing
        # features and comparing how the predictions change.
        
        
         # validation = True
        validation = False
        
        fixed_size_training = True 
        overlap_training = True 
        overlap_trade = False
        
        
        # change_mode = 'difference'
        change_mode = 'std'
        roll_window = 10
        num_points = 3
        epsilon = 0.1
        
        cols_to_change = selected_columns[:3]
        
        output_stability_file = f'{output_path_prefix}/stability_test_prob_predictions_{file_suffix}_eps={epsilon}_change_mode={change_mode}.csv'
       
        tti = TrainTest(df, model, validation, train_first_len, val_len, train_len,\
                                        test_len, label_columns, selected_columns=selected_columns,\
                                        fixed_size_training=fixed_size_training, overlap_training=overlap_training,
                                        input_width=1, label_width=1, num_days=num_days, scaling=False)
        
        tti.train_test(create_x_y_test=True, verbose=2)
        
      
        
        
        predictions_perturbed = tti.stability_test(cols_to_change, num_points, \
                               epsilon, change_mode, roll_window, filename=None)
        
            
        predictions_perturbed.loc[:,'Prediction'] = predictions_perturbed.loc[:, tti.pred_col_names].values.argmax(axis=1)     
        
        # stability_df = prob_predictions['Prediction'].groupby(level=[0,1]).apply(lambda a: is_middle_opposite(a.values))
        stability_df = predictions_perturbed['Prediction'].groupby(level=[0,1]).apply(lambda a: not_all_equal(a.values))
        
        
        stability_by_date = stability_df.groupby(level=0).sum()
        stability_by_feature = stability_df.groupby(level=1).sum()
        
        # exclude_dates = stability_by_date[stability_by_date > 0].index    
# %% Cross-validation

   
    if run_cross_validation:
        
        ##########################################################################
        # Cross-validation
        ##########################################################################
        
        
        print('\n******* Cross-validation *******:\n')
        
          # validation = True
        validation = False
        
        fixed_size_training = True 
        overlap_training = True 
        
        
        
        X_y_cross_val_file = f'{output_path_prefix}/X_y_cross_val_{file_suffix_model}.joblib'   
        
        # init TrainTest innstance (tti) if it doesn't exist
        
        if tti is None:
        
            tti = TrainTest(df, model, validation, train_first_len, val_len, train_len,\
                                                test_len, label_columns, selected_columns=selected_columns,\
                                                fixed_size_training=fixed_size_training, overlap_training=overlap_training,
                                                input_width=1, label_width=1, num_days=num_days, scaling=False,
                                                cross_val_len=cross_val_len)
            
        # cross validate tti
        
        # column names for sample weights and dates exclusion
        kwargs = {'sample_weight_col': sample_weight_col, 'exclude_index': exclude_index}
        
        tti.cross_validation(n_splits=n_splits, filename=None, verbose=2,  **kwargs)
      
        
        # tti performance
        
        position_size_col = None
        verbose=2
        
        
        
        
        performance_file = f'{output_path_prefix}/performance_cross_val_{file_suffix}.csv'
        
        performance = tti.get_performance(input_set='cross_val', delta=delta, \
                                          position_size_col=position_size_col, verbose=verbose,\
                                          filename=performance_file)
    

        # seasoanl and total performace
        df_performance_groups = tti.get_performance_groups(input_set='cross_val', delta=delta, position_size_col=position_size_col, 
                                          verbose=1)               
           
        y_set = tti.X_y_cross_val['y']
        
         
        tti.plot_performance(plot_path_prefix, input_set='cross_val')
        
        # adjusted position size 
        
        if position_from_prediction:          
             print('\nSetting position size\n')
             tti_size = copy.deepcopy(tti)
             position_size_col =  tti_size.prediction_to_pos_size(input_set='cross_val')

             performance = tti_size.get_performance(input_set='cross_val', delta=delta, \
                                              position_size_col=position_size_col, verbose=verbose,\
                                              filename=performance_file)
              
                 
             # seasoanl and total performace
             df_performance_groups = tti_size.get_performance_groups(input_set='cross_val', delta=delta, position_size_col=position_size_col, 
                                               verbose=1)                  
               
             y_set = tti_size.X_y_cross_val['y']
            
                
             tti_size.plot_performance(plot_path_prefix, input_set='cross_val')
     
             output_dic = compare_models([tti,tti_size],input_set='cross_val', path_prefix=plot_path_prefix)   

   
       
 # %% Test object       
    if run_test_object:
        
         # validation = True
        validation = False
        
        fixed_size_training = True 
        overlap_training = True 
        
        
        
        X_y_test_file = f'{output_path_prefix}/X_y_test_{file_suffix_model}.csv'  
        
        
        # create_x_y_train = False
        create_x_y_train = True
        
        
        # init TrainTest innstance (tti)
        
        tti = TrainTest(df, model, validation, train_first_len, val_len, train_len,\
                                            test_len, label_columns, selected_columns=selected_columns,\
                                            fixed_size_training=fixed_size_training, overlap_training=overlap_training,
                                            input_width=1, label_width=1, num_days=num_days, scaling=False
                                            )
        
        # train and test tti
        
        # column names for sample weights and dates exclusion
        kwargs = {'sample_weight_col': sample_weight_col, 'exclude_index': exclude_index}
        
        
       
        
        tti.train_test(verbose=1,  filename=X_y_test_file, create_x_y_train=create_x_y_train, **kwargs)      
                        
        
        # tti performance
        
        position_size_col = None
        verbose=0
        
        
        
        
        performance_file = f'{output_path_prefix}/performance_test_{file_suffix}.csv'
        
        
        
        performance_train = tti.get_performance(input_set='train', delta=delta, position_size_col=position_size_col, verbose=verbose)
                                          
            
        # performance = tti.get_performance(input_set='test', delta=delta, position_size_col=position_size_col, verbose=verbose,\
        #                                    filename=performance_file)
        
        y_set = tti.X_y_test['y']
        
        # seasoanl and total performace
        df_performance_groups = tti.get_performance_groups(input_set='test', delta=delta, position_size_col=position_size_col, 
                                          verbose=1)
        
            
        # tti.plot_performance(plot_path_prefix, input_set='test')
        
        # adjusted position size 
        
        if position_from_prediction:          
             print('\nSetting position size\n')
             tti_size = copy.deepcopy(tti)
             position_size_col =  tti_size.prediction_to_pos_size(input_set='test')

             performance = tti_size.get_performance(input_set='test', delta=delta, \
                                              position_size_col=position_size_col, verbose=verbose,\
                                              filename=performance_file)
                 
             # seasoanl and total performace
             df_performance_groups = tti_size.get_performance_groups(input_set='test', delta=delta, position_size_col=position_size_col, 
                                              verbose=1)
                   
              
               
             y_set = tti_size.X_y_test['y']
            
                
             tti_size.plot_performance(plot_path_prefix, input_set='test')
             
             comparison_file =  f'{output_path_prefix}/model_comparison.csv'
             output_dic, df_pnl_time = compare_models([tti,tti_size],input_set='test', 
                                                      filename=comparison_file,
                                                      path_prefix=plot_path_prefix)   
            # compare_models([tti,tti_1],input_set='test')              
        tti_file = None
        # tti_file = f'{output_path_prefix}/tti.joblib'
        
        if tti_file is not None:
            tti.save(tti_file)
        
        
# %% Bin model metrics
# Bin feature or target  into bins according the values of some feature 
# and calculate model metrics (performance) corresponding to each bin.
# Feature can be any column in self.df, not necessarily from self.selected_columns.

    if bin_metrics:
        input_set = 'test' if run_test_object else 'cross_val'
        
        bins = 5
        feature= selected_columns[0] 
        # feature = None
        
        df_metrics = tti.bin_metrics(bins, input_set, feature) 
        
        print(df_metrics)            
               
# %% Partial performance for index groups
# Calculate partial performances for different groups, such as Winter-Summer

    if performance_groups:
        print('\nPartial performance:\n')
        input_set = 'test' if run_test_object else 'cross_val'
        
        df_performace_groups = tti.get_performance_groups(input_set)
        
               
                                    
       
 # %% Plot tree
 # can be applied for tree-like models only
    
    if plot_tree_decisions:
        
        ###########################################################################
        # Plot tree
        ###########################################################################       
        
        plot_tree(model)
        
# %%  Random noise
   
    if random_noise:

        ################
        # Random noise #
        ################
        #  Calculate model performance after adding some noise to the test (not train) data and compare it with the performance without noise.
       
        input_set = 'test' if run_test_object else 'cross_val'
        relative = True
        
        noise_results_file = f'{output_path_prefix}/noise_results_{file_suffix}_rel={relative}.csv' 
        cols_with_noise = tti.selected_columns
        
        print('\nPerformance without noise\n')
        noise = 0. 
        num_real = 1
        
        performance_df_clean = tti.randomised_performance(cols_with_noise, noise, num_real, 
                                                         relative, input_set)
        
        print(performance_df_clean.mean())
        
        print('\nMean performance with noise\n')
        noise = 0.5 
        num_real = 10
       
        
        performance_df = tti.randomised_performance(cols_with_noise, noise, num_real, relative,
                                                    input_set, noise_results_file)

        print(performance_df.mean())       

        
        # Stability cross-validation
        # change_mode = 'std'
        # roll_window = 10 #std dev window
        # num_points = 3   
        # epsilon = 0.1 
        # n_splits = 10
        # # train_len = 20
        # output_stability_file = f'{output_path_prefix}/stability_prob_predictions_{file_suffix}_eps={epsilon}_change_mode={change_mode}.csv'
       
        
        
        # X_train = df[:train_len]
        # y = X_train[label_columns[1]].shift(-num_days)[:-num_days]
        # # y_cont = X_train[label_columns[0]].shift(-num_days)[:-num_days]
        # X = X_train[:-num_days][selected_columns]
        
        # cols_to_change = X.columns[2:] 
        
        # prob_predictions = stability_cross_validation(X, y, model, n_splits, cols_to_change, num_points, \
        #                        epsilon, change_mode, roll_window, num_days, filename=output_stability_file)
        
          
        # th_min, th_max = 0.5 - delta, 0.5 + delta 
     
        # prob_predictions.loc[:,'Prediction']  = prob_to_label(prob_predictions.loc[:,'Prob prediction'].values, th_min, th_max)    
        
        # stability_df = prob_predictions['Prediction'].groupby(level=[0,1]).apply(lambda a: not_all_equal(a.values))
        
        # stability_by_date = stability_df.groupby(level=0).sum()
        # stability_by_feature = stability_df.groupby(level=1).sum()
        
        # exclude_dates = stability_by_date[stability_by_date > 0].index
        
        
        # # stability_df = \
        # # stability_cross_validation_performance(X, y, y_cont, model, n_splits, cols,  epsilon=epsilon, 
        # #                                        relative=relative, num_points=3, delta=delta, 
        # #                                        num_days=num_days, filename=output_stability_file)
        # # print(stability_df)
        
        # # stab_sum = stability_df.sum().sort_values(ascending=False)
         
        # # print(stability_df.sum())
        
        # # stab_sum_gth_one = stability_df[stability_df > 1].sum().sort_values(ascending=False)
        
        # # print(stab_sum_gth_one)        
        
        
        
        
 # %%  Clustering
    if clustering:
                
        
        #######################
        # Clustering          # 
        ####################### 
        # Cluster features in groups according to their importance
        
        X_train = df[:train_len]
        y = X_train[label_columns[1]].shift(-num_days)[:-num_days]
        y_cont = X_train[label_columns[0]].shift(-num_days)[:-num_days]
        X = X_train[:-num_days][selected_columns]
        
        clusters, importance = clustered_MDA_feature_imp(X, y, model, maxNumClusters=10, n_init=10, 
                                                       n_splits=3, num_real=1, random_state=31)
        _ = [print(f'Cluster: {key} \n{value}\n') for key, value in clusters.items()]
        print('importance:\n', importance)
        
        
        
        # Select all features from the clusters with positive importance
        
       
        best_clstr_features = []
        clstr_indices = []
        for ind in range(len(importance)):
            if importance.iloc[ind]['mean'] > 0:
                clstr_ind = int(importance.iloc[ind:ind+1].index[0][-1])
                clstr_indices.append(clstr_ind)
                best_clstr_features += clusters[clstr_ind]
                # print(clusters[clstr_ind], '\n')
                
                
        print(f'Select {len(best_clstr_features)} out of {X.shape[1]} features from clusters {clstr_indices}')
        print(f'Selected features using clusters: {best_clstr_features}')

        
        #######################
        # Second clustering   # 
        #######################  
        
        # choose cluster or set of fetures for furhter clustering
        # features_to_cluster = clusters[1]
        # maxNumClusters = min(10, len(features_to_cluster) - 1) 
        # clusters_second, importance_second = clustered_MDA_feature_imp(X[features_to_cluster], y, model, maxNumClusters=maxNumClusters, n_init=10, 
        #                                                n_splits=10, num_real=100, random_state=31)
        # _ = [print(f'Cluster: {key} \n{value}\n') for key, value in clusters_second.items()]
        # print('importance:\n', importance_second)
        
        
# %%  Remove highly correlated features in clusters
# Searching Uncorrelated List Of Variables (SULOV) method 
   
    if sulov:
        
        from featurewiz import FE_remove_variables_using_SULOV_method
        
        sulov_selected_columns = selected_columns
        
        # sulov_selected_columns = ['WTI_Brent Swap Dealers Short F+O',
        #                           'WTI_Brent Managed Money Spreading F+O',
        #                           'WTI_Brent Producer Long',
        #                           'WTI_Brent Open Interest',
        #                           'WTI_Brent Managed Money Short F+ O change',
        #                           'WTI_Brent Managed Money Short change quantile']
        
        ############################################################
        # Searching Uncorrelated List Of Variables (SULOV) method  # 
        ############################################################
        
        X_train = df[:train_len]
        y = X_train[label_columns[1]].shift(-num_days)[:-num_days]
        y_cont = X_train[label_columns[0]].shift(-num_days)[:-num_days]
        X = X_train[:-num_days][sulov_selected_columns]        
        
        numvars = X.columns
        df_train_auto = X.copy()
        df_train_auto['target'] = y
        
        uncorr_features = \
        FE_remove_variables_using_SULOV_method(df_train_auto, numvars, modeltype='Classification', 
                                               target='target', corr_limit = 0.90, verbose=2)
        
    

# %%  Make prediction

    if make_prediction:
        
        print('\nPrediction\n')
        # eff_start, eff_end = '2000-01-01', pd.to_datetime("today").strftime("%Y-%m-%d")
        eff_start, eff_end = '2000-01-01', '2021-11-08'
        
        sym_data_filename_pred = f'{path_prefix}/Input_Data/{sym_name}_{eff_start}_{eff_end}_num_days_{num_days}_cal={cal}_drop_nan=False.csv'
        
        num_prediction_days = 10
        
        
    
        # get some extra data for making prediction
        # df_pred_data = get_data(symbols, eff_start, eff_end, num_days, 
        #           cal=cal, cal2=cal2, drop_nan=True, 
        #           countries=countries,
        #           symbols_filename=symbols_filename, 
        #           ldz_forecast_filename=ldz_forecast_filename, 
        #           filename=all_data_filename,
        #           var_filename=var_filename,
        #           day_ahead=day_ahead)
        
        # or use the data you have already
        df_pred_data = df
             
            
        X_pred = df_pred_data[-num_prediction_days:][selected_columns]
        
        # # X_pred = X_pred_from_test
        
        # binary_prediction = model.predict(X_pred)
        # prediction_proba = model.predict_proba(X_pred)

        # th_min, th_max = 0.5 - delta, 0.5 + delta 
        # prediction = prob_to_label(prediction_proba[:,1], th_min, th_max)
        # df_prediction = pd.DataFrame({'prediction':prediction}, index=X_pred.index[-num_prediction_days:])
        
        
        assert tti, 'No tti object is available'
        df_prediction = tti.predict(X_pred)
       
        print(f"\nPrediction:\n {df_prediction['y']}")
       
# %%  SHAP

    if explore_shap:
        
        
        # calculate shap values
        shap_values = tti.evaluate_shap()   
        
        
        X_test = tti.X_y_test['X'][tti.selected_columns]
        
        # summary plot
        shap.summary_plot(shap_values, X_test)   
        
        # dependence plot for individual features
        feature_ind = 1
        shap.dependence_plot( feature_ind, shap_values, X_test)
        
        # feature importance based on shap values
        
        feature_importance = np.mean(np.abs(shap_values), axis=0)

        feature_importance_df = pd.DataFrame(feature_importance, index=X_test.columns, columns=['importance'])

        feature_importance_df.sort_values('importance', inplace=True)
   
        plt.figure( figsize=(12, 15))
        feature_importance_df['importance'].plot(kind='barh', color='b')
        plt.title('Feature importance: tree feature importance')
        plt.show()
        
        # plot shap contributions for a particular raw of data
        
        ind = -1
        expected_values = tti.shap_expected_values
        
        shap.force_plot(expected_values[ind], shap_values[ind,:], X_test.iloc[ind,:], matplotlib=True) 
        
        
        shap_reg_df = tti.plot_shap_reg()
        
        tti.plot_shap_time()        
        
        


# %%  Save model for deployment

    if save_deployment_model:
        
        # save model and train/test parameteres
        
        models_filename =  f'{path_prefix}/Deployment/deployment_models_{file_suffix_model}.csv'
        
        file_exist = os.path.exists(models_filename)
        if file_exist:  # if file exists already get the latest model_id and increase it by one           
            df_models = pd.read_csv(models_filename)
            model_id = df_models['model_id'].iloc[-1] + 1
            
        else:
           model_id = 1

        
        features = list(tti.selected_columns) if tti.selected_columns else \
                   list(tti.df.columns)        
       
         
        print('\nSaving model specifications for deployment\n') 
        save_model_results(models_filename, features=[features], label_columns=[tti.label_columns],  
                              model=tti.model.__class__.__name__,
                              model_parameters=[tti.model.get_params()], validation=tti.validation, 
                              train_first_len=tti.train_first_len,
                              val_len=tti.val_len, train_len=tti.train_len, test_len=tti.test_len, 
                              fixed_size_training=tti.fixed_size_training, 
                              overlap_training=tti.overlap_training,
                              delta=tti.delta, num_days=tti.num_days,
                              **tti.model_performance_total['test'], model_id=model_id)
        
        
        
        # save metadata
        
        # metadata_filename =  f'{output_path_prefix}/deployment_metadata_{file_suffix_model}.csv'
        
        # cot_filenames = [fname.replace(path_prefix, '') for fname in bbg_filenames]
        
        # save_model_results(metadata_filename,  sym_exchange=sym_exchange, sym_name=sym_name, cal=cal,
        #                    open_day=open_day, num_days=num_days, num_days=num_days, 
        #                    position_size_col=position_size_col, cot_filenames=[cot_filenames], 
        #                    coeffs=coeffs, w1=w1, w2=w2)
        
        
        
# %%  Total prediction 

    if make_total_prediction:
        
        # make prediction for all possible feature values
        # makes sense for small number of features/values
         
        feature_values = [df[feature].unique() for feature in selected_columns]
        X = pd.DataFrame(columns=selected_columns)
        for ind, values in enumerate(product(*feature_values)):
           X.loc[ind, :] = values
        
        df_total_predict = X.copy()
        model = tti.fitted_models[-1]     
        prediction = model.predict(X) 
        df_total_predict['prediction'] = prediction
        
        
# %%  Model for errors
# Create models for error based on previous predictions errors ("error model"). Using such a model generate
# a new TrainTest instance with the predictions calculated as a sum of the
# original model and the error model predictions ("sum model"). 
# Compare the original model with the "sum model".

    if model_for_errors:
        
        # append new data for prediction
        if tti.num_days == 0:
            # replace two lines below by new data for prediction
            X = tti.df[-1:]
            X.index = X.index.shift(1, freq='D')         
            
            # append new data to tti.df
            tti.df = pd.concat([tti.df, X])
            
        tti_error, tti_sum, output_dic, df_pnl_time = create_error_model(tti, k=2, 
                                                     model='linear', comparison_file=None, 
                                                     plot_path_prefix=None, make_prediction=True, verbose=0)
        
        # prediction
        prediction_next = tti_sum.set_dic['prediction_next']['y'][tti.pred_col_name]

       
# %%  Model for error as feature
# Create models for error (not  based on previous predictions errors) using new features and a new regressor
# See description of tvt.create_error_as_feature_model() for details

    if model_for_error_as_feature:
        lgbm_model = LGBMRegressor(max_depth = 30, n_estimators = 200, reg_alpha = 0.1,  reg_lambda = 0.1, random_state = 111)
        #2nd model as described in tti obj is model_for_err
        #3nd model as described in tti obj is model_err
        model_for_err = lgbm_model
                
        tti_for_err, tti_sum, df_for_err_performance_groups, df_sum_performance_groups = tvt.create_error_as_feature_model(tti, model_for_err, model_err=None, err_shift=0, 
                                                                                    calculate_performance=True, 
                                                                                    err_features=None, verbose=1)
                        

   
# %%  Class predictions and confusion matrix

    if calculate_confusion_matrix:
        input_set = 'test'
        class_prediction = tti.get_class_prediction(input_set=input_set)
        
        target = tti.set_dic[input_set]['y']['y_binary']
        class_labels = tti.fitted_models[-1].classes_
        cm = confusion_matrix(target, class_prediction, labels=class_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=class_labels)
        disp.plot()
        plt.show()
        
     
    
    
    
    

        
        
        
        
        
        
        
        