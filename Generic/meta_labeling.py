# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:52:33 2022

Meta-labeling as described in 
"Advances in Financial Machine Learning" and 
"Machine Learning for Asset Managers (Elements in Quantitative Finance)"
        by Marcos M López de Prado (Author)

@author: aossipov

The main idea:
    1. Train and test the primary model and get its predictions.
    2. Set meta lables (done by tti.create_meta_data() method), which is 1 if
     the primeary model prediction is correct or 0 otherwise.
    3. Train and test meta model
    4.  If the prediction of the meta-model is 1, then the final decision to buy or sell 
    is the decision of the primary model, if the prediction of the meta-model is 0, the final 
    decision is not to trade.
    

To use the script one needs first to save the primary tti object (the one, for 
which meta-labeling is applied).

It works both for classification and regression as a primary model.

"""

import numpy as np
import pandas as pd


# from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier


import copy 

import sys

sys.path.append('C:/tsLocal/folib/ML_tools')



from  train_validation_test import  TrainTest, compare_models, load_tti




path_prefix = '//trs-fil01/trailstone/Alexander Ossipov/Projects/Generic/'
input_path_prefix = '//trs-fil01/trailstone/Alexander Ossipov/Projects/Generic/Input_Data'
output_path_prefix = '//trs-fil01/trailstone/Alexander Ossipov/Projects/Generic/Output_Data'
plot_path_prefix = '//trs-fil01/trailstone/Alexander Ossipov/Projects/Plots/'


    
def prob_to_position_size(x, delta=0):
    y = abs(x - 0.5)
    if delta > 0:
        y = (y / delta) ** 2
    
    return y    
   



##############################################################################
##############################################################################

if __name__ == '__main__':

# %% General parameters
    
   
    tti_primary_filename = f'{output_path_prefix}/tti.joblib' 
    
    tti_primary = load_tti(tti_primary_filename)
    
    df, label_columns = tti_primary.create_meta_data()    
    
    df.dropna(inplace=True)
    
 
    num_days = 1

   
    position_size_col = None
   
    
    train_first_len = 1
    val_len = 1
    
    # train_len = 300
    # test_len = 75j
    
    train_frac = 0.8
    val_frac = 0.
    
    # train_len = int(len(df) * train_frac)
    train_len = 375
    val_len = int(len(df) * val_frac)
    # test_len = int(len(df_bbg) * (1 - train_frac - val_frac)) - 3
    test_len = 25
    
    
    delta=0.01 # threshold for converting probability prediction to label
    
    file_suffix = f'meta'
    file_suffix_model = f'meta'
    
    # test_date = date[num_days - test_len:]
    
    
# %% Models
    
    # lgbm_model = LGBMClassifier(max_depth = 10, n_estimators = 10, lambda_l1 = 0.5,  lambda_l2 = 0.5, random_state = 111)
    
    # lgbm_model_new = LGBMClassifier(max_depth = 5, n_estimators = 50,  min_samples_leaf=0.05, lambda_l1 = 1,  lambda_l2 = 1, random_state = 111)    

    # tree_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=1, random_state=566)
    
    # rf_model = RandomForestClassifier(max_depth=3, n_estimators=60, min_samples_leaf=0.05, random_state=566)

    clf = DecisionTreeClassifier(criterion='entropy',
                                max_features=1,  
                                min_weight_fraction_leaf=0, random_state=566)

    clf = BaggingClassifier(base_estimator=clf, 
                      n_estimators=500, 
                      max_features=1., 
                      max_samples=1., 
                      oob_score=False, random_state=111)


    # model = lgbm_model
    # model = lgbm_model_new
    # model = tree_model
    # model = rf_model
    model = clf

        
    
    
    # delta=0.01 # threshold for converting probability prediction to label
    
    # n_splits = 10
    
    tti = None #  train test instance 

# %% Features    
    
    # correlations of features with the "meta lable" target
    # can help to choose the best features
    corrs = df.corrwith(df['meta_label']).sort_values()
    print(corrs)
    # selected_columns = list(corrs.index[-7:-2])
    selected_columns = tti_primary.selected_columns
    
    
# %% Control flow

    # # greedy_search = True
    # greedy_search = False

    # # bin_stat_single_target = True
    # bin_stat_single_target = False
    
    # # bin_stat = True
    # bin_stat = False
    
    # # explore_stability_cross_val = True
    # explore_stability_cross_val = False
    
    # # explore_stability_test = True
    # explore_stability_test = False
    
    # # run_cross_validation = True
    # run_cross_validation = False
    
    run_test_object=True
    # run_test_object=False
    
    # # plot_tree_decisions = True
    # plot_tree_decisions = False
    
    # # random_noise = True
    # random_noise = False

    # # clustering = True
    # clustering = False
     
    # # sulov = True
    # sulov = False
    
    # # compare_prices = True
    # compare_prices = False
    
    # # make_prediction = True
    # make_prediction = False
    
    # # explore_shap = True
    # explore_shap = False
    
    # # save_deployment_model = True 
    # save_deployment_model = False
    
        
 # %% Test object       
    if run_test_object:
        
         # validation = True
        validation = False
        
        fixed_size_training = True 
        overlap_training = True 
        
        
        
        X_y_test_file = f'{output_path_prefix}//X_y_test_{file_suffix_model}.csv'   
        
        # init TrainTest innstance (tti)
        
        tti = TrainTest(df, model, validation, train_first_len, val_len, train_len,\
                                            test_len, label_columns, selected_columns=selected_columns,\
                                            fixed_size_training=fixed_size_training, overlap_training=overlap_training,
                                            input_width=1, label_width=1, num_days=num_days, scaling=False
                                            )
        
        # train and test tti
        
        tti.train_test(verbose=1,  filename=X_y_test_file, create_x_y_train=True)       
        
        # tti performance
        
        position_size_col = None
        verbose=0
        
        
        
        
        performance_file = f'{output_path_prefix}//performance_test_{file_suffix}.csv'
        
        performance_train = tti.get_performance(input_set='train', delta=delta, position_size_col=position_size_col, verbose=verbose)
                                          
            
        
        performance = tti.get_performance(input_set='test', delta=delta, position_size_col=position_size_col, verbose=verbose,\
                                          filename=performance_file, decision_method='meta_label')
        
            
        # seasoanl and total performace
        df_performance_groups = tti.get_performance_groups(input_set='test', delta=delta, position_size_col=position_size_col, 
                                          verbose=1)
            
        
        y_set = tti.X_y_test['y']
        
            
        tti.plot_performance(plot_path_prefix, input_set='test')  
        
        
        print('\nSetting position size\n')
        
        tti_size = copy.deepcopy(tti)
        
                        
       
        tti_size.df.loc[tti_size.index.test_total_y, 'Reg Position Size'] =\
            prob_to_position_size(tti.X_y_test['y']['Prob_prediction'], delta=0.02)
    
        position_size_col = 'Reg Position Size'
                
        performance = tti_size.get_performance(input_set='test', delta=delta, position_size_col=position_size_col, verbose=verbose,\
                                         filename=performance_file, decision_method='meta_label')
        
        
        # seasoanl and total performace
        df_performance_groups = tti_size.get_performance_groups(input_set='test', delta=delta, position_size_col=position_size_col, 
                                          verbose=1)    
            
            
        y_set = tti_size.X_y_test['y']
        
            
        tti_size.plot_performance(plot_path_prefix, input_set='test')
        
        # comparing the primary and meta models
        
        print('\nComparing the primary and the meta-model\n')
        
        comparison_file =  f'{output_path_prefix}/model_comparison.csv'
        output_dic, df_pnl_time = compare_models([tti_primary, tti],input_set='test', 
                                                 filename=comparison_file,
                                                 path_prefix=plot_path_prefix)   
  
                