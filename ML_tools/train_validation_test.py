# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:37:16 2021

@author: aossipov


Module containing functions for training, validating and testing.
The main interface class is TrainTest; the objects of this class are used 
in scripts for training/testing models.

"""


import numpy as np 
import pandas as pd 

import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score, precision_score 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from statsmodels.graphics.tsaplots import plot_acf


import sys, os
import shutil
import contextlib


sys.path.append('C:/tsLocal/folib/ML_tools')

from data_windowing import train_test_split, list_from_dataset


import shap

from joblib import dump, load
from copy import deepcopy


# import tsg.core.Dates as Dates
import tsg.core.Strings as Strings
import tsg.utils.tsVisualize as tsv


class TrainTestIndex():
    """
    Auxiliary object cantaining information about index for train, cross-validation and test sets.
    """
    
    def __init__(self, train=None, val=None, test=None, cross_val=None):
    
        self.train = [] if train is None else train 
        self.val = [] if val is None else val 
        self.test = [] if test is None else test
        self.cross_val = [] if cross_val is None else cross_val 
        
        self.train_total_X,  self.train_total_y = None, None
        self.val_total_X,  self.val_total_y = None, None
        self.test_total_X, self.test_total_y = None, None
        self.cross_val_total_X, self.cross_val_total_y = None, None
         
                
        
class TrainTest():
    """
    An object of the class can be trained and tested using rolling windows. 
    After that all kind of metrics (model performance) can be calculated and visualised. 
    An object contains all the information about train and test parameters, train and test data, fitted models,
    features, predictions and perfomance.
    """
    def __init__(self, df, model, validation, train_first_len, val_len, train_len, test_len, 
                      label_columns, selected_columns=None, fixed_size_training=False, 
                      overlap_training=True,  input_width=1, label_width=1, num_days=1, scaling=False,\
                      prediction_type='classification', trading=True, model_name=None, cross_val_len=None,
                      suppress_warnings=False, pred_col_name=None):
        """

        Parameters
        ----------
        df : DataFrame
            features and additional data
        model : regressor or classifier
            model to be trained
        validation : boolean
            if True, only one set for validation is generated
        train_first_len : int
            length of the first train set
        val_len : int
            length of validation set
        train_len : int
            length of train sets
        test_len : int
            length of test sets
        label_columns : list(str1, str2)
            names of columns in df for continuous (str1) and binary labels (str2) 
        selected_columns : str or list, tuple, set of str, optional
            Names of columns in df to be used as features. 
            If None, selected_columns = df.columns 
            The default is None.
        fixed_size_training : boolean, optional
            if True, all train sets have the same length, 
            otherwise train sets are expanding   
            The default is False.
        overlap_training : boolean, optional
            if True, train sets overlap, test sets follow one after another without gaps 
            The default is True.
        input_width : int, optional
            Length of input: X(t), X(t - 1), ..., X(t - input_width + 1) is an input at time t. 
            The default is 1.
        label_width : int, optional
            Length of label: y(t), y(t + 1), ..., y(t + label_width - 1) is a lable at time t. 
            The default is 1.
        num_days : int, optional
            The time offset to generate label from the label column: y(t) = label_column(t + num_days)  
            The default is 1.
        scaling : boolean, optional
            If True, the standard scaler is applied to an input.
            The default is False.
        prediction_type : str, optional
            'classification' or 'regression' 
            The default is 'classification'.
        trading : boolean, optional
            If True, the model is of trading type, i.e. PnL and related metrics can be calculated 
            The default is True.
        model_name : str, optional
            Name of the model used in plots.  
            The default is None.
        cross_val_len : int
             length of cross validation set. If None, cross_val_len = train_len    
             The default is None.
        suppress_warnings : boolean, optional
            If True, warnings are suppressed
            The default is True.    
        pred_col_name : str, optional    
            For multi-class classification determines probability of which class
            should be used for decidions.
            The default is None.
        Returns
        -------
        None.

        """
        
        if selected_columns is None:
            selected_columns = list(df.columns)
            
        if isinstance(selected_columns, str):
            selected_columns = [selected_columns]      
    
        if not isinstance(selected_columns, list):
            selected_columns = list(selected_columns)  

        self.df = df
        self.model = model
        self.validation = validation
        self.train_first_len = train_first_len  # not used currently
        self.val_len = val_len  # not used currently
        self.train_len = train_len 
        self.test_len = test_len 
        self.cross_val_len = cross_val_len
        self.label_columns = label_columns
        self.selected_columns = selected_columns
        self.fixed_size_training = fixed_size_training
        self.overlap_training = overlap_training
        self.input_width= input_width
        self.label_width = label_width
        self.num_days = num_days
        self.scaling = scaling
        self.prediction_type = prediction_type  # 'classification' or 'regression'
        self.trading  = trading  # True for trading strategy, False otherwise
        self.model_name = model_name  # name used e.g. in plots, file etc
        self.suppress_warnings = suppress_warnings
        
        # attributes set by other methods
        self.fitted_models = []  # list of fitted models
        self.scalers = []  # list of scalers
        self.index = TrainTestIndex()
        self.X_y_test = None
        self.X_y_train = None
        self.X_y_cross_val = None
        self.X_y_prediction = None
        # set_dic specifiy  test/validation sets        
        self.set_dic = {'test': self.X_y_test, 'train': self.X_y_train, 'cross_val': self.X_y_cross_val,\
                        'prediction': self.X_y_prediction}  
        
        
        self.performance = {}  # dictionary containing DataFrames with performance for individual test/validation sets
        self.model_performance_total = {}  # dictionary containing dictionaries with total performance
        self.df_performance_groups = None  # DataFrame with seasonal and total performance
        self.model_metrics = {}  #  dictionary containing dictionaries with total model metrics
        self.delta = 0  # parameter for converting prob_prediction into model decision
        self.th_min = 0  # parameter for converting regression prediction into model decision
        self.th_max = 0  # parameter for converting regression prediction into model decision
        
        self.num_classes = len(self.df[self.label_columns[1]].unique()) if self.prediction_type == 'classification' \
                           else None  # numeber of classes for classification 
        
        self.pred_col_names = [f'Prob_prediction_{i}' for i in range(self.num_classes)] if self.prediction_type == 'classification' else\
                             ['Reg_prediction']
                             
        self.pred_col_name = pred_col_name if pred_col_name else self.pred_col_names[-1]
        
        self.target_col_name = 'y_binary' if self.prediction_type == 'classification' else\
                               'y_cont' 
       
        # parameters required to calculate performance 
        self.performance_parameters = None                 
       
        self.sample_weight_col = None  # column name for sample weights  
        self.original_sample_weight_col = None  # copy of the above columns name used to exclude dates from train set  
        self.train_dates_excluded = False  # True if some dates are excluded from train set  

        if self.num_days == 0 and self.trading:
            print('\n WARNING: Paramater trading was set to False, as num_days=0.\n')
            self.trading = False    
                               
        pd.set_option('display.float_format', '{:.2f}'.format)    
   
    
    ##############################################################################    
    def train_test(self, create_x_y_test=True,  create_x_y_train=True,  filename=None, 
                   verbose=0, **kwargs):
        """
        
        Train and test models for multiple train/test sets generated by rolling window.

        Parameters
        ----------
        create_x_y_test : boolean, optional
            if True, X_y_test is created 
            The default is True.
        create_x_y_train : boolean, optional
            if True, X_y_train is created 
            The default is True.
        filename : str, optional
            If not None, self.X_y_test['X'] and self.X_y_test['y'] are saved in files.
            The default is None.
        verbose : int, optional
           verbosity mode
           The default is 0.
           
        **kwargs : additional keywords
            
        sample_weight_col : str
            column name comtaining train weights  
            
        exclude_index : DatetimeIndex
            index to exclude from training
        
        
        Returns
        -------
        None.

        """
       
        
        
        selected_and_label_columns = self.selected_columns.copy()
        for col in self.label_columns:
            if col not in selected_and_label_columns:
                selected_and_label_columns.append(col)
                
        # adjust train and test length so that len(y_test) = test_len and
        #  len(y_train) = train_len. The y_sets generated by train_test_split
        # should have larger length in  order to compensate for eleminating rows
        # when creating label for num_days > 0 and input_width > 1
        train_len =  self.train_len + self.num_days + self.input_width - 1 
        test_len = self.test_len + self.num_days + self.input_width - 1        
        
        
        # generator for train/test sets
        # print('before split')
        split_gen = train_test_split(self.df[selected_and_label_columns], self.train_first_len, self.val_len, 
                                     train_len, test_len, self.label_columns, self.fixed_size_training, 
                                     self.overlap_training, input_width=self.input_width,
                                     label_width=self.label_width, shift=self.num_days, verbose=verbose)
        
        # print('after split')
              
        if not self.validation:
              _ = next(split_gen)
        
             
        pred_test_total = np.array([]).reshape(0, self.num_classes) if self.prediction_type == 'classification' \
                          else np.array([])  # probability prediction for classification or prediciton for regression  for test set
        pred_train_total = np.array([]).reshape(0, self.num_classes) if self.prediction_type == 'classification' \
                          else np.array([])  # the same for trian set
        
        
        y_test_binary_total = np.array([])   
        y_test_cont_total = np.array([])   
        
        y_train_binary_total = np.array([])   
        y_train_cont_total = np.array([])   
        
         
        self.index.train = []
        self.index.val = []
        self.index.test = []
    
     
        
        # sapmple weights
        self.sample_weight_col = kwargs.get('sample_weight_col', self.sample_weight_col)   
               
        
        # exclude train dates
        exclude_index = kwargs.get('exclude_index', None)   
        self.exclude_train_dates(exclude_index)  # this method may modify sample_weight_col and hence sample_weight_data below
        
        sample_weight_data = self.df[self.sample_weight_col] if self.sample_weight_col is not None else None
        
        
        # loop over different train/test sets
        for w in split_gen:
        
            X_train, y_train_binary, y_train_cont, X_test, y_test_binary, y_test_cont = X_y_from_wgen(w)
           
    
            # print(X_train.shape, y_train.shape)
            
    
          
            ##############################################
            # Exclude some columns, if required
            # this is the only way not to include features used to generate
            # labels in the train/test sets
            
            # the sipmple way below doesn't work
            # if self.selected_columns:
            #     X_train = X_train[self.selected_columns]
            #     X_test = X_test[self.selected_columns]
            l = len(selected_and_label_columns)
            m = len(self.selected_columns)
            if l > m:
                mask = [k % l <= (m - 1) for k in range(l * self.input_width)] 
                X_train = X_train[:, mask]
                X_test = X_test[:, mask]
            
           
            if self.scaling:
                ##############################################
                # Standard scaling
        
        
                # use X_train to scale X_train and X_test
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                self.scalers.append(deepcopy(scaler))
        
            
            
            if verbose > 1:
               print(X_train.shape, y_train_binary.shape, X_test.shape, y_test_binary.shape)
    
            ##############################################
            # Model train
            if verbose > 1:
                print(f'\n Train dates: {(w.train_first_ind, w.train_last_ind)} \n')
                
            
            X_train_index, y_train_index = self.get_set_index(w.train_first_ind, w.train_last_ind)
            
            # for input_width > 1 y_train_index, rather than  X_train_index, is correct index for sample weights
            sample_weight = sample_weight_data[y_train_index] if sample_weight_data is not None else None

            if self.prediction_type == 'classification':
                self.model.fit(X_train, y_train_binary, sample_weight) 
            else:
                self.model.fit(X_train, y_train_cont, sample_weight)            
            
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                self.fitted_models.append(deepcopy(self.model))
    
    
    
            ##############################################
            # Model predictions for train and test sets
            if verbose > 1:
               print(f'\n Test dates: {(w.test_first_ind, w.test_last_ind)} \n')
            
            
            # pred_test = self.model.predict_proba(X_test)[:,1] if self.prediction_type == 'classification' else\
            #             self.model.predict(X_test)
                                    
            
            # pred_train = self.model.predict_proba(X_train)[:,1] if self.prediction_type == 'classification' else\
            #             self.model.predict(X_train)
            
            pred_test = self.model.predict_proba(X_test) if self.prediction_type == 'classification' else\
                        self.model.predict(X_test)
                                    
            
            pred_train = self.model.predict_proba(X_train) if self.prediction_type == 'classification' else\
                        self.model.predict(X_train)
        
        
    
            ##############################################
            # Concatenate results with the previous ones
            
    
            y_test_binary_total = np.concatenate((y_test_binary_total, y_test_binary))
            y_test_cont_total = np.concatenate((y_test_cont_total, y_test_cont))           
            pred_test_total = np.concatenate((pred_test_total, pred_test))
            
            y_train_binary_total = np.concatenate((y_train_binary_total, y_train_binary))
            y_train_cont_total = np.concatenate((y_train_cont_total, y_train_cont))           
            pred_train_total = np.concatenate((pred_train_total, pred_train))
           
            
            self.index.train.append((w.train_first_ind, w.train_last_ind))
            self.index.val.append((w.val_first_ind, w.val_last_ind))
            self.index.test.append((w.test_first_ind, w.test_last_ind))
            
           
        
          
          
            if self.validation: 
                break
        
        # end of loop
        
        
        self.get_total_index() 
        
        # create X_y_test DataFrame with model prections
        if create_x_y_test: 
            if verbose > 0:
                print('\nCreating X_y_test set\n')
            self.X_y_test = self.create_X_y_set(self.index.test, self.index.test_total_X,\
                                               self.index.test_total_y, pred_test_total, \
                                                   y_test_binary_total, y_test_cont_total, verbose=verbose)
            
            self. set_dic['test'] = self.X_y_test 
            
            
        # create X_y_train DataFrame with model prections
        if create_x_y_train: 
            if verbose > 0:
                print('\nCreating X_y_train set\n')
            self.X_y_train = self.create_X_y_set(self.index.train, self.index.train_total_X,\
                                               self.index.train_total_y, pred_train_total, \
                                                   y_train_binary_total, y_train_cont_total, verbose=verbose)
            
            self. set_dic['train'] = self.X_y_train     
       
        
        if filename is not None:
          # dump(self.X_y_test, filename)  # best way to do it, but  
          # there is a compatibility issue between pandas 1.1.x and 1.0.x
          
          if 'test' in filename:
              test_X_file = filename.replace('test', 'test_X')
              test_y_file = filename.replace('test', 'test_y')
          elif '.' in filename:
              test_X_file = filename.replace('.', '_X.')
              test_y_file = filename.replace('.', '_y.')
          else:
              'WARNING: Data is not saved. Change the file name.'
              return
              
          try:
              self.X_y_test['X'].to_csv(test_X_file)
          except:
              print(f'WARNING: file {test_X_file} already exists and can\'t be modified.')
             
          try:    
              self.X_y_test['y'].to_csv(test_y_file)   
          except:
              print(f'WARNING: file {test_y_file} already exists and can\'t be modified.')    
          
        
        return  
        
    ##############################################################################  
    def cross_validation(self, n_splits, filename=None, verbose=0, **kwargs):
        """
        Standard k-fold cross-validation.
    

        Parameters
        ----------
        n_splits : int
            number of splits
        filename : str, optional
            If not None, self.X_y_cross_val is saved in file.
            The default is None.
        verbose : int, optional
           verbosity mode
           The default is 0.
           
        **kwargs : additional keywords
            
        sample_weight_col : str
            column name comtaining train weights  
            
        exclude_index : DatetimeIndex
            index to exclude from training
        
        Returns
        -------
        None.

        """
                                     
        cross_val_len = self.cross_val_len if self.cross_val_len is not None else self.train_len
        
        assert n_splits * (self.num_days + self.input_width) < cross_val_len,\
        f'WARNING: {n_splits=} is too big for the cross-validation length={cross_val_len}'
        
        X_train = self.df[:cross_val_len]
        last_ind = len(X_train) - self.num_days  # better than -self.num_days, as it also works for self.num_days=0
        
        y_binary = X_train[self.label_columns[1]].shift(-self.num_days)[:last_ind]
        y_cont = X_train[self.label_columns[0]].shift(-self.num_days)[:last_ind]
        y = y_binary if self.prediction_type == 'classification' else y_cont
            
        X = X_train[:last_ind][self.selected_columns]
        
        cv_generator = split_generator(l=len(X), n_splits=n_splits, num_days=self.num_days)
        
        # print(X.shape, y.shape)
        # print(X.iloc[0], y.iloc[0])
        if verbose > 0:
            print(f'Loop over {n_splits} train/test splits')
            
        pred_test_total = np.array([]).reshape(0, self.num_classes) if self.prediction_type == 'classification' \
                          else np.array([])  # probability prediction for classification or prediciton for regression  for test set
     
                
        y_test_binary_total = np.array([]) 
        y_test_cont_total = np.array([])  
      
        # position_size_test_total = np.array([]) 
        
        self.index.cross_val = []
        
        # sapmple weights
        self.sample_weight_col = kwargs.get('sample_weight_col', self.sample_weight_col)   
       
        # exclude train dates
        exclude_index = kwargs.get('exclude_index', None)   
        self.exclude_train_dates(exclude_index)  # this method may modify sample_weight_col and hence sample_weight_data below
        
        sample_weight_data = self.df[self.sample_weight_col] if self.sample_weight_col is not None else None
        
    
      
        for i, (train, test) in enumerate(cv_generator):  # loop over different train/test splits
            if verbose > 0:
                print('i =', i)
            X_train, y_train, = X.iloc[train,:], y.iloc[train] 
            X_test, y_test_binary, y_test_cont = X.iloc[test, :], y_binary.iloc[test],  y_cont.iloc[test].values
            if verbose > 0:
                print(f'Train length: {len(train)}, test length: {len(test)}')
            
            # if input_width > 1 we need to rearrange data appropriately 
            X_train = self.convert_window_input(X_train)
            X_test = self.convert_window_input(X_test)
            y_train = y_train[self.input_width - 1:]
            y_test_binary = y_test_binary[self.input_width - 1:]
            y_test_cont = y_test_cont[self.input_width - 1:]
            
            
            # for input_width > 1 y_train_index, rather than  X_train_index, is correct index for sample weights
            sample_weight = sample_weight_data[y_train.index] if sample_weight_data is not None else None
    
            
            self.model.fit(X_train, y_train, sample_weight)
            
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                self.fitted_models.append(deepcopy(self.model))
            
            # pred_test = self.model.predict_proba(X_test)[:,1] if self.prediction_type == 'classification' else\
            #             self.model.predict(X_test)
            
            pred_test = self.model.predict_proba(X_test) if self.prediction_type == 'classification' else\
                        self.model.predict(X_test)
            
            ##############################################
            # Concatenate results with the previous ones
    
            y_test_binary_total = np.concatenate((y_test_binary_total, y_test_binary))
            y_test_cont_total = np.concatenate((y_test_cont_total, y_test_cont))       
            pred_test_total = np.concatenate((pred_test_total, pred_test))
            
            self.index.cross_val.append((X.index[test[0]], X.index[test[-1]]))
          
            ##############################################
        
        # self.index.cross_val = test_ind
        self.index.cross_val_total_X, self.index.cross_val_total_y = self.get_total_set_index(self.index.cross_val)
        # create X_y_cross_val DataFrame with model prections
        
        
        # create X_y_cross_val DataFrame with model prections
        if verbose > 0:
            print('\nCreating X_y_cross_val set\n')
        self.X_y_cross_val = self.create_X_y_set(self.index.cross_val,\
                                                 self.index.cross_val_total_X,\
                                                 self.index.cross_val_total_y,\
                                                 pred_test_total, y_test_binary_total,\
                                                 y_test_cont_total, verbose=verbose)
      
        self.set_dic['cross_val'] = self.X_y_cross_val
    
       
        # if self_file is not None:
        #   dump(self, tti_file) 
        
        
        if filename is not None:
          dump(self.X_y_cross_val, filename)     
        
        
        return 
       
    ##############################################################################    
    def get_performance(self, input_set='test', verbose=2, **kwargs):
        """
        
        Calculate individual model performance for each set in input_set and the total performance.
        
        First model decisions are calculated using self.get_decision(). 
        Performance includes different model metrics calculated in self.get_model_metrics().
        If self.trading is True, then PnL related metrics calculated additionally by day_profit(). 
        

        Parameters
        ----------
        input_set : str, optional
            Name of the set, for which performance is calculated, e.g.
            'test', 'train', 'cross_val_', etc.
            The default is 'test'.
        verbose : int, optional
           verbosity mode
           The default is 0.
           
        **kwargs : additional keywords
            some parameters used as arguments in self.get_decision()
            others are additional performance parameters:
                
        position_size_col : str. optional
            Name of the position size column.
            The default is None.
        overlap_trade : boolean, optional    
            If False, then for num_days > 1 the model performance is 
            calculated as an average performance for num_days independent trades
            opened at day, day + 1, day + (num_days - 1). Otherwise, the 
            trades will be overlapping and not independent. The scenario is 
            currently not supported.
            The default is False.
        confluent_pos : boolean, optional
            If True, then the same repeating buy/sell position is considered 
            as a single position. 
            The default is True.
               
        
        Returns
        -------
        model_performance_total : dictionary
            Performance metrics and their values.

        """
        
        # save performance parameters
        self.performance_parameters = kwargs
        
        assert input_set in self.set_dic.keys(), 'input_set parameter is not valid '
        
        X_y_set = self.set_dic[input_set]
        assert X_y_set is not None, f'Create {input_set} set first'
        decision_total = self.get_decision(input_set, **kwargs)   
        
        
        if verbose > 0:
            print(f'\nPerformance for {input_set} set\n')
        
        if not self.trading:  # if not trading strategy
            # performance is determined only by model_metrics
            performance = self.get_model_metrics(input_set=input_set)  
            if verbose > 0:
                performance_str = ',  '.join([f'{key}: {performance[key]:.3f}' for key in performance])
                print(performance_str)
            # self.model_performance_total[input_set] = performance    
            return performance
        
        y_set = X_y_set['y']
        
        m = len(y_set['Set_id'].unique())
        
        position_size_col = kwargs.get('position_size_col', None) 
        overlap_trade = kwargs.get('overlap_trade', False)
        confluent_pos = kwargs.get('confluent_pos', True)
        
        y_cont_total = y_set['y_cont'] 
        position_size_total = 1 if position_size_col is None else \
                self.df[position_size_col][y_cont_total.index]
                
    
        
        
        for ind, i in enumerate(y_set['Set_id'].unique()):
    
           
            y_cont = y_set['y_cont'][y_set['Set_id']==i]      
            position_size =  1 if position_size_col is None else \
                position_size_total[y_cont.index]
            decision = decision_total[y_cont.index]
               
            model_performance = day_profit(y_cont, decision, 
                                            overlap_trade=overlap_trade, num_days=self.num_days,
                                            position_size=position_size, 
                                            confluent_pos=confluent_pos, verbose=0)
            model_performance.pop('PnL')
            
            # additional metrics
            pred = y_set[self.pred_col_names][y_set['Set_id']==i] 
            target = y_set[self.target_col_name][y_set['Set_id']==i] 
            if self.prediction_type == 'classification':
                 additional_metrics = classification_metrics(target, pred, decision, self.suppress_warnings)           
            else:            
                 additional_metrics = regression_metrics(target, pred, decision, self.num_days)
                 binary_target = binary_return(target)
                 additional_metrics.update(classification_metrics(binary_target, pred, decision, self.suppress_warnings))
    
            if ind == 0:
                # DataFrame with performance for individual test sets
                performance_cols = list(model_performance.keys()) + list(additional_metrics.keys())\
                                   + ['Start_date', 'End_date']
                
                self.performance[input_set] = pd.DataFrame(index=range(m), \
                                              columns=performance_cols)
                # self.performance[input_set].loc[:, 'Dates'] = ''
             
            
            self.performance[input_set].loc[i, model_performance.keys()] =\
                list(model_performance.values())
                
            
            self.performance[input_set].loc[i, additional_metrics.keys()] =\
                list(additional_metrics.values())  
            
                
            # self.performance[input_set].loc[i, 'Dates'] =\
            #     (y_cont.index[0], y_cont.index[-1])
            self.performance[input_set].loc[i, ['Start_date', 'End_date']] =\
                (y_cont.index[0].date(), y_cont.index[-1].date())    
           
        if verbose > 1:
            print(f'\n{self.performance[input_set]}\n')
        
       
        
        model_performance_total = day_profit(y_cont_total, decision_total,
                                            overlap_trade=overlap_trade, num_days=self.num_days,
                                            position_size=position_size_total, 
                                            confluent_pos=confluent_pos, verbose=verbose)
                     
       
        # move PnL from model_performance_total to y_set
        y_set['PnL'] = model_performance_total.pop('PnL')
        
        
        # add additional performance metrics
        
        pred = y_set[self.pred_col_names]
        target = y_set[self.target_col_name]
        decision = y_set['Decision']
        
        if self.prediction_type == 'classification':
             additional_metrics = classification_metrics(target, pred, decision, self.suppress_warnings)           
        else:            
             additional_metrics = regression_metrics(target, pred, decision, self.num_days)
             binary_target = binary_return(target)
             additional_metrics.update(classification_metrics(binary_target, pred, decision))
        
        model_performance_total.update(additional_metrics)      
        
        if verbose > 0:
            for key, value in additional_metrics.items():
                print(f'{key}: {value:.2f}', end='   ')
            print()    
        
        filename =  kwargs.get('filename', None)
        if filename is not None:
            features = list(self.selected_columns) if self.selected_columns else \
                       list(self.df.columns)
           
            performance_parameters = self.performance_parameters.copy()
            performance_parameters.pop('filename')
            
            if 'delta' in performance_parameters:
                performance_parameters.pop('delta')
                
           
            save_model_results(filename, features=[features], label_columns=[self.label_columns],  
                                model=self.model.__class__.__name__,
                                model_parameters=[self.model.get_params()], validation=self.validation, 
                                train_first_len=self.train_first_len,
                                val_len=self.val_len, train_len=self.train_len, test_len=self.test_len, 
                                fixed_size_training=self.fixed_size_training, 
                                overlap_training=self.overlap_training,
                                delta=self.delta, num_days=self.num_days, input_width=self.input_width,
                                **model_performance_total, **performance_parameters, 
                                first_date=y_set.index[0].date().strftime('%Y-%m-%d'),
                                last_date=y_set.index[-1].date().strftime('%Y-%m-%d'))
        
        self.model_performance_total[input_set] = model_performance_total  # dictionary with total performance
        
       
        
        return model_performance_total
    
    ##############################################################################    
    def get_performance_groups(self, input_set='test', method='winter_summer', verbose=1, **kwargs):
        """
        Calculate model performance for index groups (time periods).
        First create groups by calling group_index(), then calcluate 
        performance for each group using self.get_performance().
        

        Parameters
        ----------
        input_set : str, optional
            Name of the set, for which performance is calculated, e.g.
            'test', 'train', 'cross_val_', etc.
            The default is 'test'.
        method : str, optional
            Method of splitting into groups used as an argument in group_index(). 
            The default is 'winter_summer'.
            
        **kwargs : additional keywords
            performance parameters used as arguments in self.get_performance().

        Returns
        -------
        df_performance_groups : dictionary
            Performance metrics and their values for different groups and for 
            the whole set.

        """
        # save performance parameters
        self.performance_parameters = kwargs
       
        assert input_set in self.set_dic.keys(), 'input_set parameter is not valid '            
        X_y_set = self.set_dic[input_set]
        assert X_y_set is not None, f'Create {input_set} set first'           
        y_set = X_y_set['y']
        
        index_groups = group_index(y_set.index, method, **kwargs) 
        performance_list = []
        gr_label_list = []
        
        filename = kwargs.get('filename', None)
        
        if 'filename' in kwargs:
            filename = kwargs.pop('filename')             
        else:
            filename = None
        
        for gr_label, gr_index in index_groups.items():
            
            self.set_dic[f"group {gr_label}"] = {'y': y_set.loc[gr_index]}
            
    
            performance = self.get_performance(input_set=f"group {gr_label}", verbose=0, **kwargs)
            performance_list.append(performance)
            gr_label_list.append(gr_label)
            
        df_performance_groups = pd.DataFrame(performance_list, index=gr_label_list)
        
        cols_first = ['sharpe', 'sortino', 'drawdown_to_pnl', 'total PnL', 'max_drawdown']
        
        if all(col in df_performance_groups.columns for col in cols_first):  #  this is not True for trading=False
            cols_other = [col for col in df_performance_groups.columns if col not in cols_first]
            df_performance_groups = df_performance_groups[cols_first + cols_other]
    
            
        # total performance
        model_performance_total = self.get_performance(input_set=input_set, verbose=0, **kwargs)
          
        model_performance_total = pd.DataFrame(model_performance_total, index=['total'])
        df_performance_groups = pd.concat([df_performance_groups, model_performance_total])
        self.df_performance_groups = df_performance_groups
        
        if verbose > 0:
            print(df_performance_groups.iloc[:,:6])   
        
        
        if filename is not None:
            df_performance_groups.to_csv(filename)
        
        return df_performance_groups    
           
    ##############################################################################    
    def get_decision(self, input_set='test', **kwargs):
        """
        Calculate model trading decisions from model predictions.

        Parameters
        ----------
        input_set : str, optional
            Name of the set, for which performance is calculated, e.g.
            'test', 'train', 'cross_val_', etc.
            The default is 'test'
            
        **kwargs : additional keywords
           
        decision_method : str, optional
            Method of converting model prediction to trading decisions.
            The default is '3_step'.
            
            Classification decision_methods:
                '3_step' : for binary classification 
                           decision = -1, if prob_prediction < 0.5 - delta
                           decision = 1, if prob_prediction >= 0.5 + delta 
                           decision = 0, otherwise
                           
                           for multiclass classification
                           'decision_weights': list or np.array, optinal
                           weights for calculating weighted probalbility
                           weighted_prob = sum(prob_prediction(i) * decision_weights(i))
                           len(decision_weights) = num_classes
                           The default value (-w, -w + 1,..., w), where 
                           w = (num_classes - 1) / 2                           
                           decision = -1, if weighted_prob < -delta
                           decision = 1, if weighted_prob >= delta 
                           decision = 0, otherwise
                           
                'class_to_position_size' :
                           First, the class with the highest probability is determined.
                           Then the corresponding decision is taken according to 'class_decisions'.
                           'class_decisions': list or np.array, optinal
                           decision = class_decisions(i), where i is the index of the class with 
                           the highest probability.
                           len(class_decisions) = num_classes
                           The default value (-m / 2, -m / 2 + 1, ..., m / 2), if m is even
                                             (-(m - 1) / 2, -(m - 1) / 2  + 1, ..., (m + 1) / 2), if m is odd
                                             where m = num_classes
                                             
                'meta_label' :
                           Implemented only for binary classification.
                           meta_label_pred = 1, if prob_prediction for meta-model > 0.5 + delta
                           meta_label_pred = 0 , otherwise   
                           decision = (decision of primary model) * metal_label_pred
                           
              
                
              Regression decision_methods:             
                '3_step' : based on two additional otional parameters:
                           'th_min': float, optional
                           The default value is 0.
                           'th_max': float, optional
                           The default value is 0.
                           decision = -1, if regression prediction < th_min
                           decision = 1, if regression prediction >= th_max
                           decision = 0, otherwise
                           
                           
        exclude_index : index, optional
            If not None, decisions set to 0 for dates from index.
            The default is None.
            
        Returns
        -------
        y_set['Decision']  : pd.Series
            trading decisions 

        """
         
        assert input_set in self.set_dic.keys(), 'input_set parameter is not valid '
        
        X_y_set = self.set_dic[input_set]
        assert X_y_set is not None, f'Create {input_set} set first'
        
        y_set = X_y_set['y']
        
        
         
        self.decision_method = kwargs.get('decision_method', '3_step')
        
        
        # Classification
        if self.prediction_type == 'classification':
             self.delta = kwargs.get('delta', self.delta)          
             if self.decision_method == '3_step':  # default method for classification                 
                if self.num_classes == 2:                     
                    self.decision_class = kwargs.get('decision_class', self.num_classes - 1)  # label class, whose probalility determines the decision
                    prob_prediction =  y_set[f'Prob_prediction_{self.decision_class}']
                    th_min, th_max = 0.5 - self.delta, 0.5 + self.delta 
                    decision = prob_to_label(prob_prediction, th_min, th_max) 
                else:
                    m = (self.num_classes - 1) / 2
                    default_weights = np.arange(start=-m, stop=(m + 1))    
                    self.decision_weights = np.array(kwargs.get('decision_weights', default_weights))
                    prob_predictions = y_set[self.pred_col_names]
                    weighted_prob = (prob_predictions * self.decision_weights).sum(axis=1).values
                    decision = prob_to_label(weighted_prob, -self.delta, self.delta) 
             
             if self.decision_method == 'class_to_position_size':
                 label_class = y_set[self.pred_col_names].values.argmax(axis=1)
                 m = self.num_classes
                 if m % 2 == 0:
                     default_class_decisions = np.array(list(range(-m // 2, 0)) + list(range(1, (m // 2) + 1)))
                 else:
                     default_class_decisions = np.arange(-(m - 1) // 2, (m + 1) // 2)
                 self.class_decisions =  np.array(kwargs.get('class_decisions', default_class_decisions))  
                 decision = self.class_decisions[label_class]     
            
             if self.decision_method == 'meta_label':  # decision based on meta-labeling 
                prob_prediction =  y_set[self.pred_col_name]
                meta_label_pred = (prob_prediction > 0.5 + self.delta).astype(int)
                y_set['meta_label_pred'] = meta_label_pred
                primary_decision = self.df['primary_decision'][y_set.index]
                decision = primary_decision * meta_label_pred
       
        # Regression
        if self.prediction_type == 'regression':
             
             if self.decision_method == '3_step':  # default method for regression 
                 # it assumes that we predict returns
                 reg_prediction =  y_set['Reg_prediction']
                 self.th_min = kwargs.get('th_min',  self.th_min)
                 self.th_max = kwargs.get('th_max', self.th_max)
                 decision = prob_to_label(reg_prediction, self.th_min, self.th_max) 
                
                 
        y_set['Decision'] = decision
        
        exclude_index = kwargs.get('exclude_index', None)
        if exclude_index is not None:
            y_set['Decision'][ y_set.index.isin(exclude_index)] = 0     
           
        return  y_set['Decision'] 
           
    
    ##############################################################################    
    def get_class_prediction(self, input_set='test'):
        
        assert self.prediction_type == 'classification', 'get_class_prediction() method can be applied only to classification'
        
        assert input_set in self.set_dic.keys(), 'input_set parameter is not valid '
        
        X_y_set = self.set_dic[input_set]
        assert X_y_set is not None, f'Create {input_set} set first'
        
        y_set = X_y_set['y']
        
        class_labels = self.fitted_models[-1].classes_
        
        y_set['Class_prediction'] = class_labels[y_set[self.pred_col_names].values.argmax(axis=1)]
        
        return y_set['Class_prediction']
        
        
        
        
    
    ##############################################################################    
    def get_model_metrics(self, input_set='test'):
        """
        Return model metrics by calling classification_metrics() for classifiers
        or regression_metrics() for regressors.

        Parameters
        ----------
        input_set : str, optional
            Name of the set, for which performance is calculated, e.g.
            'test', 'train', 'cross_val_', etc.
            The default is 'test'
            
        Returns
        -------
        model_metrics : dictionary
        Model metrics and their values.

        """
                
        assert input_set in self.set_dic.keys(), 'input_set parameter is not valid '
        
        X_y_set = self.set_dic[input_set]
        assert X_y_set is not None, f'Create {input_set} set first'
        
        y_set = X_y_set['y']
        
        pred = y_set[self.pred_col_names]
        target = y_set[self.target_col_name]
        decision = y_set['Decision'] if 'Decision' in y_set.columns else None
        
      
        if self.prediction_type == 'classification':
             model_metrics = classification_metrics(target, pred, decision, self.suppress_warnings)           
        else:            
             model_metrics = regression_metrics(target, pred, decision, self.num_days)
           
        self.model_metrics[input_set] = model_metrics
        
        return model_metrics
          
     
    
    #############################################################################
    def predict(self, X, ind=-1, trained_in_past=False, time_delta=pd.Timedelta(0), prediciton_set_name='prediction', **kwargs):
        """
        Make model prediction and model decision.
        
        

        Parameters
        ----------
        X : DataFrame
            Input data containing features; may contain additional columns.
        ind : int, optional
            index for model in self.fitted_models 
            The default is -1.
        trained_in_past : boolean, opt
            If True, for each datapoint in X the latest model trained in past is
            used for prediction.
            The default is False.
        time_delta: pandas Timedelta object, optional
            If trained_in_past is True, represents an additonal time which must 
            pass after the last date of the train index before the model 
            can be used for prediction.
            The default is pd.Timedelta(0).
        prediciton_set_name : str, optional
            Name for the prediction set. 
            The default is 'prediction'.
            
            
        **kwargs : additional keywords
            performance parameters used as arguments in self.get_decision()

        Returns
        -------
        self.set_dic[prediciton_set_name] :  dictionary
           X and y data sets for prediction

        """
        
        assert self.fitted_models, 'No traiend models are available'
        
        X = X[self.selected_columns]  # keep only selected columns 
        
        pred_X = X.copy()  # copy input data for prediction DataFrame
        
        # if input_width > 1 we need to rearrange data appropriately 
        X = self.convert_window_input(X)
        
        pred_y = pd.DataFrame(index=pred_X.index[self.input_width - 1:] , columns=[*self.pred_col_names, 'Decision'])
        
        # scale data if required 
        if self.scaling:
            scaler = self.scalers[ind]  # use the last scaler by default
            X = scaler.transform(X)
        
        if not trained_in_past: # prediction for a given model from self.fitted_models 
            model = self.fitted_models[ind]  # use the last model by default
            
            prediction = model.predict_proba(X) if self.prediction_type == 'classification' else\
                         model.predict(X)  # make prdiction 
            
            if self.prediction_type == 'classification':
                pred_y[self.pred_col_names] = prediction
            else:
                pred_y[self.pred_col_name] = prediction
                
            pred_y['Set_id'] = ind
        
        else:  # prediction using the most recent models trained in the past
            l = len(self.fitted_models)  # number of different models 
            train_index = self.index.train
            pred_index = pred_y.index
            for ind in range(l):
                model = self.fitted_models[ind]
                mask_1 =  pred_index >= (train_index[ind][1] + time_delta)  # dates after the last train date + delta for the current model
                
                if ind < l - 1:  # more recent model is available
                    mask_2 =  pred_index < (train_index[ind + 1][1] + time_delta)  # datesbefore the last train date + delta for the next model
                    selected_index =  pred_index[mask_1 & mask_2]
                else:  # this is the latest available model
                    selected_index = pred_index[mask_1]
                
                if len(selected_index) > 0:
                    X_slice = X.loc[selected_index, :]
                    prediction_slice = model.predict_proba(X_slice) if self.prediction_type == 'classification' else\
                                  model.predict(X_slice)  
                    pred_y.loc[selected_index, 'Set_id'] = ind
                    
                    if self.prediction_type == 'classification':
                        pred_y.loc[selected_index, self.pred_col_names] = prediction_slice
                    else:
                        pred_y.loc[selected_index, self.pred_col_name] = prediction_slice
                        
            if pred_y[self.pred_col_name].isna().sum() > 0:
                print('WARNING: prediction can\'t be made for all dates')
            pred_y = pred_y[~pred_y[self.pred_col_name].isna()] 
            
            assert len(pred_y) > 0, 'No models trained in the past are available'
            
        
       
        self.X_y_prediction = {'X': pred_X, 'y': pred_y}
        
        self.set_dic[prediciton_set_name] = self.X_y_prediction
        
        self.get_decision(input_set=prediciton_set_name, **kwargs)  # create decisiona 
        
        return self.set_dic[prediciton_set_name]
     
        
     
    ##############################################################################    
    def create_X_y_set(self, date_index_list, index_total_X, index_total_y, pred, y_binary, y_cont, verbose=0):
        """
        Create a DataFrame with features and model predictions.  
        

        Parameters
        ----------
        date_index_list : list of tuples consisting of two Timestamps: [(t1, t2), ...]
            list of first and last Timesteps indices
        index_total_X : DatetimeIndex
            total datetime index for X set    
        index_total_y : DatetimeIndex
            total datetime index for y set    
        pred : one-dimensional numpy array
            model predictions
        y_binary : one-dimensional numpy array
           buinary label used for classifiacation 
        y_cont : one-dimensional numpy array
           continuous label used for regression
        verbose : int, optional
           verbosity mode
           The default is 0.

        Returns
        -------
        X_y_set : dictionary {'X': DataFrame, 'y': DataFrame}
            X_y_set['X'] contains X sets - features
            X_y_set['y'] contains corresponding y sets consisting of continuous and binary labels, 
            model predictions 

        """
        if verbose > 1:
              print('\n Creating set with predictions \n')
       
        X_y_set = {'X': pd.DataFrame(index=index_total_X, columns=self.selected_columns),\
                   'y': pd.DataFrame(index=index_total_y, columns=(*self.pred_col_names, 'y_binary', 'y_cont'))}
      
        
        X_y_set['X'] = self.df.loc[index_total_X, self.selected_columns].copy()
        
        # one can't do it like this, if different sets overlap (which is the case for train sets)
        # X_y_set['y'][self.pred_col_names] = pred
        # X_y_set['y']['y_binary'] = y_binary
        # X_y_set['y']['y_cont'] = y_cont  
        
        ind_start = 0   
        # loop over different train/test sets
        for i, (date_start, date_end) in enumerate(date_index_list):    
            if verbose > 1:
                print('Set: ', date_start, date_end)
            index_X, index_y = self.get_set_index(date_start, date_end)  
            X_y_set['X'].loc[index_X, 'Set_id'] = i   
            X_y_set['y'].loc[index_y, 'Set_id'] = i                                   
                                                             
            ind_end = ind_start + len(index_y)
            X_y_set['y'].loc[index_y, self.pred_col_names] = pred[ind_start: ind_end]
            X_y_set['y'].loc[index_y,'y_binary'] = y_binary[ind_start: ind_end]
            X_y_set['y'].loc[index_y,'y_cont'] = y_cont[ind_start: ind_end]            
            ind_start = ind_end
        
        
        
        X_y_set['y'] = X_y_set['y'].astype({**{col: float for col in self.pred_col_names}, 'y_binary': int, 'y_cont': float, 'Set_id': int})
        X_y_set['X']['Set_id'] = X_y_set['X']['Set_id'].astype(int)  
        
        if verbose > 1:        
            print('Total predictions length: ', len(X_y_set['y']))
        
        return X_y_set
    
    
    ##############################################################################    
    def plot_performance(self, path_prefix, input_set='test'):  
        """
        Plot model performance related to PnL.

        Parameters
        ----------
        path_prefix : str
            Path prefix for saving plots.
        input_set : str, optional
            Name of the set, for which performance is calculated, e.g.
            'test', 'train', 'cross_val_', etc.
            The default is 'test'.

        Returns
        -------
        None.

        """
        
        if path_prefix is None:
            path_prefix='//trs-fil01/trailstone/Alexander Ossipov/Projects/Plots/Temp/'
        
        assert input_set in self.set_dic.keys(), 'input_set parameter is not valid '
        
        X_y_set = self.set_dic[input_set]
        assert X_y_set is not None, f'Create {input_set} set first'
        
        y_set = X_y_set['y']
        
        assert 'PnL' in y_set.columns, 'PnL wasn\'t calculated for this set'
        
        path_prefix = os.path.join(path_prefix, '')  # add slash to path if required
        
        
        model_name = f'{self.model_name}// ' if self.model_name else ''
        str_title = f'{model_name}{input_set=}'
        
        i = 0 if self.prediction_type == 'regression' else 1
        target = self.label_columns[i]
        str_title += f'// {target=}'
        
        model = self.model.__class__.__name__
        str_title += f'//  {model=}' 
        
        num_features = len(self.selected_columns)
        str_title += f'//  {num_features=}' 
        
        features = ', '.join(self.selected_columns)
        num_first_features = 3
        first_features = features if num_features <= num_first_features else\
                        ', '.join(self.selected_columns[:num_first_features])
        
        position_size_col = self.performance_parameters.get('position_size_col', None)\
                            if self.performance_parameters else None
                            
        
        
        str_title += f'// {position_size_col=}'  
        
        fig_title = f'{str_title}// {first_features=}'
        
        fig_pnl, ax_pnl = plot_profit(y_set['PnL'], y_set['y_cont'], y_set['Decision'], y_set.index,
                                      title=f'PnL  {fig_title}') 
        
        figs = {f'pnl_{input_set}': fig_pnl}
        
        
       
        
        if self.df_performance_groups is not None:
            fig_performance, ax_per = tsv.plotDFtable(self.df_performance_groups, strTitle=fig_title, 
                                                      equalFormat=False)
            figs[f'performance_{input_set}'] = fig_performance
        else:
            print('\nWARNING: performance DataFrame wasn\'t generated and won\'t be saved')            
        
        # attr_list = ['test_len', 'delta']
        # attr_str = tti.attributes_to_str(attr_list)
        
        
        
        str_title += f'// {features=}' 
        save_path = f'{path_prefix}{Strings.slugify(target)}/'
        save_plots(figs, save_path, str_title=str_title)

        return
    
    ##############################################################################    
    def plot_prediction(self, path_prefix, input_set='test',  **kwargs):    
        """
        Plot model predictions.

        Parameters
        ----------
        path_prefix : str
            Path prefix for saving plots.
        input_set : str, optional
            Name of the set, for which performance is calculated, e.g.
            'test', 'train', 'cross_val_', etc.
            The default is 'test'.
 
        Returns
        -------
        None.


        """
        
        if path_prefix is None:
            path_prefix='//trs-fil01/trailstone/Alexander Ossipov/Projects/Plots/Temp/'
        
        assert input_set in self.set_dic.keys(), 'input_set parameter is not valid '
        
        X_y_set = self.set_dic[input_set]
        assert X_y_set is not None, f'Create {input_set} set first'
        
        path_prefix = os.path.join(path_prefix, '')
        
        y_set = X_y_set['y']
        
        model_name = f'{self.model_name}// ' if self.model_name else ''
        str_title = f'{model_name}{input_set=}'
        
        
        # currently implemented for regression only
        if self.prediction_type == 'regression':
              
            target = self.label_columns[0]  # regression target
            str_title += f'// {target=}'
            
            model = self.model.__class__.__name__
            str_title += f'//  {model=}' 
            
            num_features = len(self.selected_columns)
            str_title += f'//  {num_features=}' 
            
            features = ', '.join(self.selected_columns)
            num_first_features = 3
            first_features = features if num_features <= num_first_features else\
                            ', '.join(self.selected_columns[:num_first_features])
            
            
            fig_title = f'{str_title}// {first_features=}'
            
            
            fig, ax = plot_reg_prediction(y_set['y_cont'], y_set['Reg_prediction'], y_set.index,
                                          title=f'Regression prediction  {fig_title}') 
            
            figs = {f'reg_prediction__{input_set}': fig}
            
            
           
            
            if self.df_performance_groups is not None:
                fig_performance, ax_per = tsv.plotDFtable(self.df_performance_groups, strTitle=fig_title, 
                                                          equalFormat=False)
                figs[f'performance_{input_set}'] = fig_performance
            else:
                print('\nWARNING: performance DataFrame wasn\'t generated and won\'t be saved')            
            
            # attr_list = ['test_len', 'delta']
            # attr_str = tti.attributes_to_str(attr_list)
            
            
            
            str_title += f'// {features=}' 
            save_path = f'{path_prefix}{Strings.slugify(target)}/'
            save_plots(figs, save_path, str_title=str_title)
            
            return
    ###############################################################################
    def plot_scatter_prediction(self, feature, input_set='test'):
        """
        

        Parameters
        ----------
        feature : str
            feature to be used for X-axis
        input_set : str, optional
            Name of the set, for which performance is calculated, e.g.
            'test', 'train', 'cross_val_', etc.
            The default is 'test'.

        Returns
        -------
        None.

        """
        
        assert input_set in self.set_dic.keys(), 'input_set parameter is not valid '
        X_y_set = self.set_dic[input_set]
        assert X_y_set is not None, f'Create {input_set} set first'
        
        y_set = X_y_set['y']
        X_set = X_y_set['X']
        
        y_predict = y_set[self.pred_col_name]
        y_target = y_set[self.target_col_name]
        X = X_set[feature] 

        plt.scatter(X, y_target, c='r', label='target') 
        # plt.plot(X, y_predict)
        plt.scatter(X, y_predict, c='b', label='prediction')
        plt.xlabel(feature)
        plt.ylabel('target')
        plt.legend()
        plt.show()
        
    ###############################################################################
    def plot_error_acf(self, input_set='test'):
        """
        Plot error autocorrelation function.

        Parameters
        ----------
        input_set : str, optional
            Name of the set, for which performance is calculated, e.g.
            'test', 'train', 'cross_val_', etc.
            The default is 'test'.
 
        Returns
        -------
        None.

        """
        
        assert input_set in self.set_dic.keys(), 'input_set parameter is not valid '
        X_y_set = self.set_dic[input_set]
        assert X_y_set is not None, f'Create {input_set} set first'
        
        y_set = X_y_set['y']
        error = y_set[self.target_col_name] - y_set[self.pred_col_name] 
        y_set['error'] = error
        title = 'Probability error autocorrelation' if self.prediction_type == 'classification'\
                else 'Error autocorrelation'
        plot_acf(error, title=title)
        
        if self.prediction_type == 'classification':
        
            decision_error = 2 * y_set['y_binary'] - 1 - y_set['Decision'] 
            y_set['decision error'] = decision_error
            plot_acf(decision_error, title="Decision error autocorrelation")
        
        plt.show()
        return
    
    
    
    ##############################################################################  
    def mean_target_return(self, by_day=False, by_day_year=False,
                           by_day_month=False, by_day_month_year=False,
                           binary=False):
        """
        Returns mean of the absolute value, median of the absolute value and 
        std of target.
        
       

        Parameters
        ----------
        by_day : boolean, optional
            If True, return separate statistics for each day of week.
            The default is False.
        by_day_year : boolean, optional
            If True, return separate statistics for each day of week and year.
            The default is False.
        binary : boolean, optional
            If True, statistics is calculated for the binary target.
            The default is False.
        by_day_month : boolean, optional
            If True, return separate statistics for each day of month.
            The default is False.
        by_day_month_year : boolean, optional
            If True, return separate statistics for each day of month and year.
            The default is False. 

        Returns
        -------
        if by_day or byday_year or by_day_month or by_day_month_year:
            stat_df  : DataFrame
            Table with the corresponding statistics.
            
        m1, m2, m3 : float, float, float  
            Mean of the absolute value, median of the absolute value, 
            standard deviation 

        """
        
        target_col = self.label_columns[1] if binary else self.label_columns[0]
        
    
        
        if by_day:
           
            self.df['day'] = self.df.index.day_of_week
            self.df[f'abs {target_col}'] = self.df[target_col].abs()
            stat_df = self.df.groupby('day')[[target_col, f'abs {target_col}']].mean()
            self.df.drop(columns=['day', f'abs {target_col}'])
            return stat_df
       
        if by_day_year:
           
            self.df['day'] = self.df.index.day_of_week
            self.df['year'] = self.df.index.year
            self.df[f'abs {target_col}'] = self.df[target_col].abs()
            stat_df = self.df.groupby(['year', 'day'])[[target_col, f'abs {target_col}']].mean()
            stat_df
            self.df.drop(columns=['day', 'year',  f'abs {target_col}'])
            return stat_df
         
        if by_day_month:
           
            self.df['day_month'] = self.df.index.day
            self.df[f'abs {target_col}'] = self.df[target_col].abs()
            stat_df = self.df.groupby('day_month')[[target_col, f'abs {target_col}']].mean()
            self.df.drop(columns=['day_month', f'abs {target_col}'])
            return stat_df
       
        if by_day_month_year:
           
            self.df['day_month'] = self.df.index.day
            self.df['year'] = self.df.index.year
            self.df[f'abs {target_col}'] = self.df[target_col].abs()
            stat_df = self.df.groupby(['year', 'day_month'])[[target_col, f'abs {target_col}']].mean()
            stat_df
            self.df.drop(columns=['day_month', 'year',  f'abs {target_col}'])
            return stat_df   
        
        

        return self.df[target_col].abs().mean(),  self.df[target_col].abs().median(), \
               self.df[target_col].std()           
    
        
    #############################################################################
    def greedy_feature_selection(self, cols, n_splits,
                             reverse=True, select_metric='sortino', 
                             verbose=0, feature_rank=False,  position_size_col=None, 
                                 n_features_penalty=0, **kwargs):
        '''
            Feature selection by greedy search.
    
            Parameters
            ----------
            cols : list 
                features to explore by greedy search
            n_splits :int
               Number of splits in cross-validation
            reverse : boolean, optional
                If True, features are removed by greedy search.
                The default is True.
            select_metric : str, optional
                Metric to maximise 
                The default is 'sortino'.
            verbose : int
                verbosity mode
                The default is 0.    
            feature_rank : boolean, optional
                If True, additionaly to best features returns model performance
                when individually features are eliminated
                The default is False.
            position_size_col : str, optional
                Column name containing position size. 
                The default is None.
            n_features_penalty : float, optional
                A feature is removed if 
                new_performance > (1 - n_features_penalty) * previous_performance 
                The default is 0.
                
            **kwargs : more optional paramters
                
                key : function, optional
                function that serves as a key for the prefromance comparison 
                The default is None.
    
            Returns
            -------
            None.
    
       '''
        
        id_func = lambda x: x
        
        func = kwargs.get('key', id_func)
            
        cols = list(cols)
        cols_copy = cols.copy()
        
        
        # cross validate 
        self.cross_validation(n_splits=n_splits, verbose=0)
         
        # performance            
        performance = self.get_performance(input_set='cross_val',  \
                                          position_size_col=position_size_col, \
                                          verbose=verbose, **kwargs)
        
        best_performance = performance[select_metric]
            
           
        if verbose > -1:
            print(f'Best performance before loop: {select_metric}  {best_performance}')
            # bp = {'mean_profit': mean_profit, 'sharpe': sharpe, 'sortino': sortino}
            # print(f'Best performance before loop: {bp}')
            
        
        exclude_columns = []
        i = 1 
        while cols:
            if verbose > 0:
                print(f'Loop over features, run {i}')
                
                
            performance_list = self.feature_remove_performance(cols, n_splits,  
                                                            exclude_columns=exclude_columns,
                                                             verbose=verbose,
                                                            position_size_col=position_size_col,**kwargs) 
                                              
               
            
            metric_list = [x[select_metric] for x in performance_list]
            max_performance = max(metric_list, key=func)
            if i==1:
                initial_performance_list = performance_list
                
            if func(max_performance) >= func(best_performance) - n_features_penalty * abs(func(best_performance)):
                best_performance = max_performance
                if verbose > -1:
                    print(f'Current best performance: {select_metric} {best_performance}')
                max_ind =  metric_list.index(max_performance)
                removed_col = cols.pop(max_ind)
                exclude_columns.append(removed_col)
                
                #check that there will be no empty set of selected columns at the next step
                if len(cols) == 1 and set(self.selected_columns) == set(exclude_columns).union(cols):
                    break
                if verbose > -1:
                    print(f'Removed feature {removed_col}')
                    print(f'Remaining features to check  {cols}')
                i += 1
            else:
                break   
                
        
        removed_cols =  [x for x in cols_copy if x not in cols]
        if not removed_cols:
            print('No features were removed')
        else:
            print(f'\nFinal best performance: {select_metric} {best_performance}')
            print('All removed features: ', removed_cols)
            # print('cols: ', cols)
        
        if feature_rank:
            return removed_cols, initial_performance_list
        
        return removed_cols    
    
    
    ##############################################################################
    def feature_remove_performance(self,  cols, n_splits, 
                                   exclude_columns=None, verbose=0,
                                   position_size_col=None, **kwargs):
        """
        Return list of model cross-validation performances calculated for
        featuters build from cols with one col excluded.

        Parameters
        ----------
        cols : list 
            features to explore by greedy search
        n_splits :int
           Number of splits in cross-validation
        exclude_columns : str, list, tuple or set, optional
            Columns from exclude_columns are not included as features. 
            The default is None.
        verbose : int
            verbosity mode
            The default is 0.    
        position_size_col : str, optional
            Column name containing position size. 
            The default is None.
        **kwargs : additional keywords
            performance parameters used as arguments in self.get_performance()

        Returns
        -------
        performance_list : TYPE
            DESCRIPTION.
            
      
        """
        
        if exclude_columns is None:
            exclude_columns = []
        
        if not isinstance(exclude_columns, (list, tuple, set)):
            exclude_columns = [exclude_columns] 
      
    
        performance_list = []
        
        selected_columns_copy = self.selected_columns.copy()
    
        for col in cols:
            print(col)
    
            self.selected_columns = [c for c in selected_columns_copy if c not in exclude_columns + [col]]
                 
            # cross validate 
            self.cross_validation(n_splits=n_splits, verbose=0)
             
            # performance
                      
            performance = self.get_performance(input_set='cross_val', \
                                              position_size_col=position_size_col, \
                                              verbose=verbose, **kwargs)
                                                  
            # if 'PnL' in performance:
            #     performance.pop('PnL')
          
                
            performance_list.append(performance)
            
        self.selected_columns = selected_columns_copy    
        
        return performance_list
    
         
    ###############################################################################
    # Add some noise to the test data (not to the train) and get perfomnce
    # cols -- columns for which noise is added
    def randomised_performance(self, cols, noise, num_real, relative=True, input_set='test', filename=None,
                               **kwargs):
        """
        
        Return model performance after adding some noise to the test (not train) data.

        Parameters
        ----------
        cols : list like
            list of column names for which noise is added
        noise : int, list, tuple or set
            Noise strength. If  noise is int, the noise strength is the same 
            for all columns, ohterwise it is column specific 
            and len(noise) = len(cols)
        num_real : int
            number of realisations of random noise
        relative : boolean, optional
            If True, the actual noise strength = noise * std of the change of the
            corresponding column, otherwise noise strength = noise
            The default is True.
        input_set : str, optional
            Name of the set, for which performance is calculated, e.g.
            'test', 'train', 'cross_val_', etc.
            The default is 'test'
        filename : str, optional
            If not None, file name for saving output.
            The default is None.
        
       **kwargs : additional keywords
           performance parameters used as arguments in self.predict() and self.get_performance()

        Returns
        -------
        performance_df : DataFrame
            model pereformance 

        """
    
   
        m = len(cols)
        
        assert input_set in self.set_dic.keys(), 'input_set parameter is not valid '
        X_y_set = self.set_dic[input_set]
        assert X_y_set is not None, f'Create {input_set} set first'
        
        X = X_y_set['X']
        y_set = X_y_set['y']
        l = len(X)
        
        if not isinstance(noise, (list, tuple, set)):
                noise = noise * np.ones(m)
        
        if relative:        
            X_change = X - X.shift(1)
            X_change[:1] = X_change[1:2]  # copy the second row to the first one to get rid of NaNs and keep the size   
            noise_strength = noise * X_change[cols].std().values    
        else:
            noise_strength = noise
        
        X_random = X.copy() 
        
        num_models = len(self.fitted_models)
        
        performance_list = []
        
        for i in range(num_real):
            # print(i)
            random_noise = np.random.uniform(-0.5, 0.5, size=(l, m)) *  noise_strength
            X_random[cols] = X_random[cols] + random_noise
            
            y_set_list = []
            for ind in range(num_models):
                X_ind = X_random[X_random.Set_id == ind]
                prediction_set = self.predict(X_ind, ind, **kwargs)
                y_set_list.append(prediction_set['y'])
                
            y_set_prediction = pd.concat(y_set_list)
            missing_cols = [col for col in y_set.columns if col not in y_set_prediction.columns]
            y_set_prediction[missing_cols] = y_set[missing_cols]
            
            X_y_prediction = {'X': X_random, 'y':  y_set_prediction}
            
            self.set_dic['noise_prediction'] =  X_y_prediction
            
            performance_list.append(self.get_performance(input_set='noise_prediction', verbose=0,  **kwargs))    
            
        performance_df = pd.DataFrame(performance_list)
        
        if filename is not None:
            performance_df.to_csv(filename)
        return performance_df
       
    
    ##############################################################################
    def stability_test(self, cols_to_change, num_points, \
                           epsilon, change_mode, roll_window, input_set='test', filename=None):
        """
        
        Model predictions for the test, cross-validation or other set perturbed by some noise.

        Parameters
        ----------
        cols_to_change : list of str
            Columns to perturb 
        num_points : int
            number of perturbation points
            if num_points is odd the middle point is the unperturbed one  
        epsilon : int or list of int
            strength of perturbation, can be different for different columns
        change_mode : str
            'difference' : epsilon is multiplied by the absolute difference of
            the column change w.r.t. to the previous row
            'std' :  epsilon is multiplied by std of the column from a rollig window 
                    of size roll_window 
            'absolute' :  epsilon is the same for all rows  
        roll_window : int
            size of a rollig window for change_mode='std'
        input_set : str, optional
            Name of the set, for which medel predictions are calculated, e.g.
            'test', 'train', 'cross_val_', etc.
             The default is 'test'.    
        filename : str, optional
            If not None, file name for saving output.
            The default is None.
            

        Returns
        -------
        predictions_perturbed  : DataFrame
            predictions for the perturbed test set
            Index of predictions_perturbed is Multiindex as created by create_perturbed_df()
            For regression it contains regression predictions for perturbed feature values,
            for classification it contains probability predections for perturbed feature values.

        """
 
        X_y_set = self.set_dic[input_set]
        assert X_y_set is not None, f'Create {input_set} set first'
        
        X = X_y_set['X'][self.selected_columns]
                   
        assert self.fitted_models, 'No fitted models found'
    
             
        
        if self.X_y_prediction is not None and input_set == 'test':  # include predictions if they are available
            additional_index = self.X_y_prediction['X'].index[~self.X_y_prediction['X'].index.isin(X.index)]
            if len(additional_index) > 0 :
                X = pd.concat([X, self.X_y_prediction['X'].loc[additional_index, :]])
        
        X_perturbed = create_perturbed_df(X, cols_to_change, num_points, epsilon, change_mode, roll_window)
       
        X_perturbed.dropna(inplace=True)
        
        X_perturbed.sort_index(level=[0], inplace=True)
          
        predictions_perturbed =  pd.DataFrame(index=X_perturbed.index)
        
        num_sets = len(X_y_set['X']['Set_id'].unique())
        for ind, i in enumerate(X_y_set['X']['Set_id'].unique()):
            
        # for ind, (test_start, test_end) in enumerate(self.index.test):
            
             model = self.fitted_models[i]
             slice_index = X_y_set['X'][X_y_set['X']['Set_id']==i].index.intersection(X.index)
               
             X_slice = X_perturbed.loc(axis=0)[slice_index, :, :]
             # for the last ind we include predictions if they are available
             if (ind == num_sets - 1) and (self.X_y_prediction is not None)\
                 and (len(additional_index) > 0) and input_set == 'test':
                 X_perturbed_pred = X_perturbed.loc(axis=0)[additional_index, :, :]
                 X_slice = pd.concat([X_slice, X_perturbed_pred])
                 slice_index = slice_index.union(additional_index)
             
             X_slice = self.convert_window_input(X_slice)
             
             # scale data if required 
             if self.scaling:
                 scaler = self.scalers[i]  
                 X_slice = scaler.transform(X_slice)
                          
             pred_slice = model.predict_proba(X_slice) if self.prediction_type == 'classification' else\
                           model.predict(X_slice)  
                          
             
             if self.prediction_type == 'classification':
                 predictions_perturbed.loc[slice_index, self.pred_col_names] = pred_slice
             else:
                 predictions_perturbed.loc[slice_index, self.pred_col_name] = pred_slice
             
             
        predictions_perturbed.dropna(inplace=True)    
                  
                                  
        return predictions_perturbed     
            
    
    ##############################################################################    
    # Bin feature or target  into bins and calculate model metrics corresponding to each bin. 
    # If feature=None, bin y_cont, if cont_target=True or y_ninary if y_cont=False
    
    def bin_metrics(self, bins, input_set='test', feature=None, cont_target=True, quantiles=True, **kwargs):  
        """
        Bin feature or target  into bins according the values of some feature 
        and calculate model metrics (performance) corresponding to each bin.
        Feature can be any column in self.df, not necessarily from self.selected_columns.
        If feature=None, bin y_cont, if cont_target=True or y_binary otherwise.

        Parameters
        ----------
        bins : int or list-like of float (if quantiles=True)
             or sequence of scalars, or IntervalIndex (if quantiles=False)
             The criteria to bin by. T
             The same as bins argument in pd.qcut() (if quantiles=True) or
             in pd.cut() (if quantiles=False).
        input_set : str, optional
            Name of the set, for which performance is calculated, e.g.
            'test', 'train', 'cross_val_', etc.
             The default is 'test'.
        feature : str, optional
            Name of the feature to be segmented into bins. 
            If None, target is segmented into bins.
            The default is None.
        cont_target : boolean, optional
            If True, y_cont is used as aa target, otherwise y_binary.   
            The default is True.
        quantiles : boolean, optional
            If True, segemntation into equal-sized buckets based on rank or based
            on sample quantiles, as in pd.qcut(). Otherwise binning is the same
            as in pd.cut().
            The default is True.
            
        **kwargs : additional keywords
            performance parameters used as arguments in self.get_performance().

        Returns
        -------
        df_metrics : DataFrame
           Model metrics for each bin.

        """
   
        if feature is None:
           df_input = self.set_dic[input_set]['y']
           col = 'y_cont' if cont_target else 'y_binary'
           feature_values = df_input[col]
        else:
           assert feature in self.df.columns, f'Input {feature=} is not in columns' 
           feature_values = self.df.loc[self.set_dic[input_set]['X'].index, feature]
           col = feature
           
    
        y_set = self.set_dic[input_set]['y']
    
        num_values =  len(feature_values.unique())
        assert num_values != 1,  f'WARNING: Column {col} contains a single value only and will be ignored'
    
        if isinstance(bins, int) and num_values < bins:
            print(f'WARNING: The number of bins will be reduced to {num_values}, \
                  which is the number of unique values for column {col}.')
            # intervals = feature_values
            # intervals_unique = np.sort(intervals.unique())
            bins = num_values
        # else:  
            
        intervals = pd.qcut(feature_values, bins, duplicates='drop') if quantiles else\
                    pd.cut(feature_values, bins=bins)
        intervals_unique = intervals.unique().sort_values()            
        
        
        len_array =  np.array([])
        index_dic = {}

        for interval in intervals_unique:
            index = y_set.index.intersection(intervals[intervals == interval].index)
            # the length below is calculated incorrectly, as it seems there is the following bug in pandas:
            # the interval.left and interval.right are rounded, so that I get wrong result for enquality below
            # For example, feature_values[0] = 0.133333333, and it is assigned to the interval (0.1, 0.133],
            # but feature_values[0] <= interval.right is False, because  interval.right =0.133 instead of
            # 0.133333333
            # Explanation of the bug: see precision parameter in cut/qcut which has the defaul value = 3
            #l = len(feature_values[(feature_values > interval.left) & (feature_values <= interval.right)])
            l = len(index)
            len_array = np.append(len_array, l)
            
            index_dic[interval] = index

        len_array = np.append(len_array, len(y_set))    
        
          
        df_metrics = self.get_performance_groups(input_set=input_set, method='from_index_dic',
                                                index_dic=index_dic, verbose=0, **kwargs)

        df_metrics['Fraction'] = len_array / len(y_set) 
        df_metrics.index.name = feature
 
        return df_metrics
    
    
    ############################################################################## 
    def pnl_bin_stat(self, input_set='test', bins=5, quantiles=True, q_min=0.1, q_max=0.9):
        """
        Bin 'PnL' into bins then calculate mean
        features in self.df w.r.t. that bins.
        Normalise the result by subtracting mean(feature) and dividing by
        std(feature).
        Additionaly calculate fraction of points between quatiles q_min and
        q_max for a given PnL bin relative to the total fraction of points
        between quatiles q_min and q_max.
        
        Return a DataFrame with the corresponding statistics.
       
        Parameters
        ----------
        input_set : str, optional
            Name of the set, for which performance is calculated, e.g.
            'test', 'train', 'cross_val_', etc.
            The default is 'test'.      
        bins : int or array
            (Number of) bins. See pd.cut for more info.
            The default value is 5.
        quantiles: boolean
            If True use pd.qcut for quantile-based discretization,
            otherwise use pd.cut to bin values. 
            The default is True.
        q_min: float
            Min quantile
        q_max: float
             Max quantile    
            


        Returns
        -------
        df_stat_mean : DataFrame
            DataFrame with statistics of each feature w.r.t. PnL bins.

        """
        

        assert input_set in self.set_dic.keys(), 'input_set parameter is not valid '
        X_y_set = self.set_dic[input_set]
        assert X_y_set is not None, f'Create {input_set} set first'

        y_set = X_y_set['y']
        df =  y_set[['PnL']].join(self.df, how='inner')   


        # df_stat = get_bin_stat(df, 'PnL', bins, quantiles)[0]
        
       
        target_col = 'PnL'
        
        df_stat_mean = target_feature_bin_stat(df, target_col, bins=bins, quantiles=quantiles, q_min=q_min, q_max=q_max)  
        
        return df_stat_mean
    
    ############################################################################## 
    def error_bin_stat(self, input_set='test', bins=5, quantiles=True, q_min=0.1, q_max=0.9):
        """
        Bin error between prediction and target for regression into bins then calculate mean
        features in self.df w.r.t. that bins.
        Normalise the result by subtracting mean(feature) and dividing by
        std(feature).
        Additionaly calculate fraction of points between quatiles q_min and
        q_max for a given PnL bin relative to the total fraction of points
        between quatiles q_min and q_max.
        
        Return a DataFrame with the corresponding statistics.
       
        Parameters
        ----------
        input_set : str, optional
            Name of the set, for which performance is calculated, e.g.
            'test', 'train', 'cross_val_', etc.
            The default is 'test'.      
        bins : int or array
            (Number of) bins. See pd.cut for more info.
            The default value is 5.
        quantiles: boolean
            If True use pd.qcut for quantile-based discretization,
            otherwise use pd.cut to bin values. 
            The default is True.
        q_min: float
            Min quantile
        q_max: float
             Max quantile    
            

        Returns
        -------
        df_stat_mean : DataFrame
            DataFrame with statistics of each feature w.r.t. error bins.

        """
        

        assert self.prediction_type == 'regression', 'The method can be applied only to regression'            

        X_y_set = self.set_dic[input_set]
        assert X_y_set is not None, f'Create {input_set} set first'

        y_set = X_y_set['y'].copy()
        y_set['error'] = y_set['y_cont'] - y_set['Reg_prediction']
        df =  y_set[['error']].join(self.df, how='inner')   
        
       
        target_col = 'error'
        
        df_stat_mean = target_feature_bin_stat(df, target_col, bins=bins, quantiles=quantiles, q_min=q_min, q_max=q_max)  
        
        return df_stat_mean
    
    
    
    ##############################################################################    
    def exclude_train_dates(self, exclude_index=None):
        """
        Exclude some data from the train set based on index. When called the second time with
        exclude_index=None, restore the data in the train set.

        Parameters
        ----------
        exclude_index : DataFrame.index, optional
            If not None, the data from the train set with index inexclude_index 
            is excluded. Otherwise the data is restored in the train set.
            The default is None.

        Returns
        -------
        None.

        """
        if exclude_index is not None:
            
            self.train_dates_excluded = True
            if self.original_sample_weight_col is None:
                self.original_sample_weight_col = self.sample_weight_col  
            weight = (~self.df.index.isin(exclude_index)).astype(int)
            if self.sample_weight_col is None:        
                self.df['sample_weight_exclude_index'] = weight
            else:
                self.df['sample_weight_exclude_index'] = weight * self.df[self.sample_weight_col]
                
            self.sample_weight_col = 'sample_weight_exclude_index'  
         
        elif self.train_dates_excluded:
            self.sample_weight_col = self.original_sample_weight_col
            self.train_dates_excluded = False
        
        return
      
    ##############################################################################    
    def prediction_to_pos_size(self,  input_set='test', transform='abs_normalised'):
        """
        Convert prediction for returns into position size and insert it into 
        self.df.

        Parameters
        ----------
        input_set : str, optional
            Name of the set, for which performance is calculated, e.g.
            'test', 'train', 'cross_val_', etc.
             The default is 'test'.
        transform : str, optional
            Name of the transformation.
            'abs_normalised' - position size is determined by the absolute
            value of the prediction.
            The default is 'abs_normalised'.
     

        Returns
        -------
        position_size_col : str
            Name of the position size column.

        """
        
        y = self.set_dic[input_set]['y'][self.pred_col_name]
        
        th = 0 if self.prediction_type == 'Regression' else 0.5
      
        if transform == 'abs_normalised':
            pos_size = abs(y - th) 
            # normalise positions, so that <abs(pos_size)> = 1
            # pos_size /= pos_size.mean()
            
            
        self.df.loc[y.index, 'Pred Position Size'] = pos_size
        position_size_col =  'Pred Position Size'      
              
           
        return  position_size_col
    
    ############################################################################## 
   
    def price_to_return(self, input_set='test', current_price=None):
        """
        Convert price predicitons to return predictions.

        Parameters
        ----------
        input_set : str, optional
            Set name: 'test', 'cross_val', 'train' or other  
            The default is 'test'.
            
        current_price: Series or one-dimensional numpy array
            Price values
            

        Returns
        -------
        None.

        """
        
        assert input_set in self.set_dic.keys(), 'input_set parameter is not valid '
        
        X_y_set = self.set_dic[input_set]
        assert X_y_set is not None, f'Create {input_set} set first'
        
        y_set = X_y_set['y']
        
        assert current_price is not None, 'current_price should be proveided as an argument'
        
        y_set['Reg_prediction']  -= current_price
        y_set['y_cont']  -= current_price
        y_set['y_binary'] = 0  # plays no role anyway
        
        return
   
    ##############################################################################
    def evaluate_shap(self, selected_index=None):
        """
        Evaluate SHAP values for the test set or its subset.

        Parameters
        ----------
     
        selected_index : DataFrame.index, optional
            If not None, specify a subset of the test set to be used for evaluating SHAP values.
            The default is None.

        Returns
        -------
        np.array()
            SHAP values

        """
        
        assert self.X_y_test is not None, 'Create X_y_test set first'
        
        # X_train = self.df[:self.train_len][self.selected_columns]
        # X_train = self.convert_window_input(X_train)
        # X_train_med = X_train.median().values.reshape((1,X_train.shape[1]))
        
        X_train = self.df[:self.train_len][self.selected_columns]
        X_train = self.convert_window_input(X_train)
        X_train_med = np.median(X_train, axis=0)
        X_train_med = X_train_med[np.newaxis,...]
        
        
        model_list = self.fitted_models
        
        l = X_train.shape[1]
        shap_values_total = np.empty(shape=[0, l])
        expected_values = np.array([])
        num_models = len(model_list)

        
        # for (date_start, date_end), model in zip(date_index_list, model_list):
        print(f'Calculating shap values for {num_models} test sets:')   
        
        X_test_total = self.X_y_test['X']
        
        if selected_index is not None:
            X_test_total = X_test_total.loc[selected_index, :]    
            
        for i, model in enumerate(model_list):    
            print(f'Test set {i}')
            
            
            X_test =  X_test_total[X_test_total['Set_id']==i][self.selected_columns]
            X_test = self.convert_window_input(X_test)
            
            explainer = shap.KernelExplainer(model.predict_proba, X_train_med) \
                if (self.prediction_type == 'Classification') else  shap.KernelExplainer(model.predict, X_train_med)
                   
                
                
            shap_values = explainer.shap_values(X_test)
            exp_values = explainer.expected_value
            
            if (self.prediction_type == 'Classification'):
                shap_values = shap_values[1]
                exp_values = exp_values[1]
            
            shap_values_total = np.concatenate((shap_values_total, shap_values))
            expected_values = np.append(expected_values, exp_values)
            
        self.shap_values = shap_values_total
        self.shap_expected_values = expected_values
        
        return self.shap_values
    
    
    ###############################################################################
    def plot_shap_reg(self, features=None):
        """
        Plot SHAP value vs. feature value for each feature in features.
        Additinaly fit the results with linear and quadratic regression.
        SHAP values should be evaluated before calling the method.
        

        Parameters
        ----------
        features : list, optional
            List of features. If None, features = self.selected_columns
            The default is None.

        Returns
        -------
        shap_reg_df : DataFrame
             Vaues if regression coefficients.

        """
        
        assert self.shap_values is not None, 'Evaluate SHAP first'
        
        if features is None:
            features = self.selected_columns
        
        micolumns = pd.MultiIndex.from_tuples([('Linear', 'x'), ('Linear', 'c'), ('Linear', 'R^2'), \
                                            ('Quadratic', 'x^2'), ('Quadratic', 'x'), ('Quadratic', 'c'),\
                                            ('Quadratic', 'R^2')], names=["reg1", "reg2"])

        
        shap_reg_df = pd.DataFrame(index=features,  columns= micolumns)
                                  
        
        for feature in features:
            
            feature_ind = self.selected_columns.index(feature)
            feature_shap_values = self.shap_values[:, feature_ind]
            feature_values = self.X_y_test['X'][feature].values.reshape(-1, 1)
            
            x, y =  feature_values,  feature_shap_values
            
            # Linear regression
            
            l_reg = LinearRegression().fit(x, y)
    
            a1, b1, Rsqrd1 = l_reg.coef_[0], l_reg.intercept_, l_reg.score(x, y)
            
            shap_reg_df.loc[feature, ('Linear',)] = a1, b1, Rsqrd1
            
            # Quadratic regression
            
           
            model = Pipeline([('poly', PolynomialFeatures(degree=2)),  \
                              ('linear', LinearRegression(fit_intercept=False))])
              
            model = model.fit(x, y) 
            
            x_ = model.named_steps['poly'].transform(x)
            (c2, b2, a2), Rsqrd2 = model.named_steps['linear'].coef_,  model.named_steps['linear'].score(x_, y)
            
            shap_reg_df.loc[feature, ('Quadratic',)] = a2, b2, c2, Rsqrd2  
            
            # plot x, y and regression
            
            plt.scatter(x, y)
            x_pred = np.linspace(x.min(), x.max()).reshape(-1,1)
            
            y_pred = l_reg.predict(x_pred)
            plt.plot(x_pred, y_pred)
            
            y_pred = model.predict(x_pred) 
            plt.plot(x_pred, y_pred)
            
            # plt.text(0,75, 0.75, f'R^2 linear reg: {Rsqrd1:.2f}\nR^2 quadratic reg: {Rsqrd2:.2f}',  transform=transAxes)
            plt.xlabel('feature value')
            plt.ylabel('SHAP')
            plt.title(feature)
            plt.show()  
            
        return shap_reg_df       

    
    ###############################################################################    
    def plot_shap_time(self, features=None):
        """
        Plot SHAP values and feature values vs. index (time) for each feature in features.
        SHAP values should be evaluated before calloing the method.

        Parameters
        ----------
        features : list, optional
            List of features. If None, features = self.selected_columns
            The default is None.

        Returns
        -------
        None.

        """
        
        assert self.shap_values is not None, 'Evaluate SHAP first'
        
        if features is None:
            features = self.selected_columns
        
        for feature in features:
            
            feature_ind = self.selected_columns.index(feature)
            feature_shap_values = self.shap_values[:, feature_ind]
            feature_values = self.X_y_test['X'][feature].values
            
            
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.suptitle(feature)
            
            ax1.plot(self.index.test_total_X, feature_shap_values, 'o')
            ax1.set_ylabel('SHAP')
            
            ax2.plot(self.index.test_total_X, feature_values, 's')
            ax2.set_ylabel('feature value')
            
            plt.show()
            
        return
    
    
    ###############################################################################                              
    def create_meta_data(self, from_test_only=False, filename=None, **kwargs):
        """
        Create DataFrame with meta-labels based on the decisions of the model.

        Parameters
        ----------
        from_test_only : boolean, optional
            If True, only the test set is used to generate meta-labels, 
            otherwise both the train and test sets. 
            The default is False.
            
            
       **kwargs : additional keywords
           performance parameters used as arguments in self.get_decision()
           
           
        Returns
        -------
        df : DataFrame
            Original self.df with new columns, containing meta-labels,
            original decisions, target and prediction.
        label_columns : list
            Column names for original continuous label and meta-label.

        """
        
        pred_col = self.pred_col_name  
        if 'Decision' not in  self.X_y_test['y'].columns:
            self.get_decision('test', **kwargs)      
            
        df_test = self.X_y_test['y'][[pred_col, 'y_binary', 'Decision']].copy()
       
        if from_test_only:          
            df_meta =  df_test 
        else:
            if 'Decision' not in  self.X_y_train['y'].columns:
                self.get_decision('train', **kwargs)      
                
            df_train = self.X_y_train['y'][[pred_col, 'y_binary', 'Decision']].copy()
            df_meta = df_test.combine_first(df_train)
         
        th_min = 0.5 if self.prediction_type == 'classification' else 0
        th_max = th_min        
        df_meta['primary_binary_pred'] = (prob_to_label(df_meta[pred_col], th_min, th_max) + 1) // 2  
        
        # df = self.df.loc[df_meta.index].copy() 
        df = self.df.copy() 
        df = df.join(df_meta['primary_binary_pred'], how='outer')
        if 'y_binary' in df.columns:
            df.drop(columns=['y_binary'], inplace=True)
        df = df.join(df_meta['y_binary'], how='outer').rename(columns={'y_binary': 'primary_y_binary'})
        if 'Decision' in df.columns:
            df.drop(columns=['Decision'], inplace=True)
        df = df.join(df_meta['Decision'], how='outer').rename(columns={'Decision': 'primary_decision'})
        df = df.join((df_meta['primary_binary_pred'] == df_meta['y_binary']).astype(int).rename('meta_label'), how='outer')
        df['meta_label'] = df['meta_label'].shift(self.num_days)
        label_columns = [self.label_columns[0], 'meta_label'] 
        
        if filename is not None:
            print(f'\nSaving meta-data to file: {filename}\n')
            meta_dict = {'df': df, 'label_columns': label_columns} 
            dump(meta_dict, filename) 
        
        return df, label_columns     
       
      
    #############################################################################  
    def attributes_to_str(self, attr_list):
        """
        Auxiliary method returning a string with object attributes and their values.
    
        Parameters
        ----------
        attr_list : list of str
            list of obect attributes [attr1, attr2, ...]
    
        Returns
        -------
        output_str : str
            f'_attr1={attr1}_attr2={attr2}_...'
    
        """
           
        attr_dic = vars(self)
        output_str = ''    
       
        for attr in attr_list:
            if attr not in attr_dic:
                print(f'\WARNING object has no attribute {attr}')
                continue
            attr_value = attr_dic.get(attr, None)
            if attr_value is None:
                print(f'\WARNING attribute {attr} is None')
                continue
            output_str += f'_{attr}={attr_value}'  
           
        return output_str    
           

                

                 
   ##############################################################################
    def get_set_index(self, first_ind, last_ind):
        """
        Retruns index for X and y sets

        Parameters
        ----------
        first_ind : Timestamp 
            first index 
        last_ind : Timestamp
            last index (not included)

        Returns
        -------
        index_X : DatetimeIndex
            datetime index for X set    
        index_y : DatetimeIndex
            datetime index for y set   

        """
        index_X, index_y = None, None
        if first_ind != -1:
            index_X = self.df[first_ind: last_ind].index
            index_y = self.df[first_ind: last_ind].index[self.input_width - 1: ]  # for y we skip first input_width - 1 rows
        return index_X, index_y    
        
    
    def get_total_set_index(self, index_list):
        """
        Retruns total datetime index for X and y sets

        Parameters
        ----------
        index_list : list of tuples consisting of two Timestamps: [(t1, t2), ...]
            list of first and last Timesteps indices

        Returns
        -------
        index_X : DatetimeIndex
            total datetime index for X set    
        index_y : DatetimeIndex
            total datetime index for y set   


        """
        
        first_ind, last_ind = index_list[0]
        index_X, index_y = self.get_set_index(first_ind, last_ind)
                        
        if len(index_list) > 1:
            for first_ind, last_ind  in index_list[1:]:
                    if first_ind != -1:
                        new_index_X, new_index_y = self.get_set_index(first_ind, last_ind)
                        index_X = index_X.union(new_index_X)
                        index_y = index_y.union(new_index_y)
        return index_X, index_y    
        
             
    def get_total_index(self):
        """        
        Generate total index for train, validation and test sets.

        Returns
        -------
        None.

        """
        self.index.train_total_X, self.index.train_total_y  = self.get_total_set_index(self.index.train)
        self.index.val_total_X, self.index.val_total_y = self.get_total_set_index(self.index.val)
        self.index.test_total_X, self.index.test_total_y = self.get_total_set_index(self.index.test)
        return
        
    ##############################################################################    
   
    def convert_window_input(self, X):
        """
        Auxiliary method to convert df or array into correct array for input_width > 1.
        If X is n x m array, then output is (n - input_width + 1) x (m * input_width) array


        Parameters
        ----------
        X : DataFrame or numpy array
            Features

        Returns
        -------
            numpy array
            transformed features for input_width > 1

        """
        
        
        if self.input_width == 1:
            return X
        
        v = X.values if isinstance(X, pd.DataFrame) else X 
            
        new_v = []
        for i in range(len(v) - self.input_width + 1):
            new_v.append(v[i: i + self.input_width, :].reshape(1,-1))
            
        new_v = np.vstack(new_v)
        
        return new_v
 
    
    ##############################################################################    
    def save(self, tti_file, verbose=1):
        """
        Dump tti object in joblib file.

        Parameters
        ----------
        tti_file : str
           file name
        verbose : int
            verbosity mode
            The default is 1.    

        Returns
        -------
        None.

        """
    
        if verbose > 0:
            print(f'\nSaving TrainTest object to file: {tti_file}\n')
        dump(self, tti_file) 
   
        
        
###############################################################################
###############################################################################


 
class DeterministicModel(): 
    """
    Abstract class defining deterministic models, for which no training is required.
    This interface allows one to use them as an input for TrainTest objects.
    The methods of the class are  suiatble both for Regressors and Classifiers.
    
    """
    
    def __init__(self, func): 
        """
        func is a function that maps 2-dim input numpy array X to 1-dim prediction
        numpy array y. In case, of Classifier y is prediction probability rather
        than prediction.

        Parameters
        ----------
        func : function
            Functuion which defines model prediction.

        Returns
        -------
        None.

        """
       
        
        self.func = func
        self.X = None
        
    def fit(self, X, y, sample_weight=None):
        pass
    
    def get_params(self):
        return {}
    
    def check_input_shape(self, X):
    # check the input array shape and copy X to self.X    
        if isinstance(X, (pd.Series, pd.DataFrame)):
            self.X = X.values
        else:
            self.X = X
            
        # check shape            
        if len(X.shape) != 2:
            print('Expected 2D array, got 1D array instead.')
            print('Reshape your data either using array.reshape(-1, 1)\
                  if your data has a single feature or array.reshape(1, -1) \
                      if it contains a single sample.')
            
            return False
        
        return True                        

    
class DeterministicRegressor(DeterministicModel): 
    """
    Child class for determenistic Regressors.
    """
    
    def predict(self, X):  
        
        if not self.check_input_shape(X):
            return
            
        return self.func(self.X)

    
class DeterministicClassifier(DeterministicModel):
    """
    Child class for determenistic Classifiers.
    """
    
    def  predict_proba(self, X):
        if not self.check_input_shape(X):
            return
            
        return self.func(self.X)
        
    
    def predict(self, X):  # standard method to get prediction from probability
        if not self.check_input_shape(X):
            return
        
        pred_prob = self.func(self.X)
        return (pred_prob > 0.5).astype(int)
    
    
def create_deterministic_model(func, model_type='Regressor'):
    """
    Retrun DeterministicRegressor() or DeterministicClassifier() defined
    by function func.

    Parameters
    ----------
    func : function
        Functuion which defines model prediction.
    model_type : str, optional
        'Regressor' or 'Classifier'
        The default is 'Regressor'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if model_type == 'Regressor':
        return DeterministicRegressor(func)
    return DeterministicClassifier(func)
    



###############################################################################
###############################################################################

def plot_profit(profit, target, prediction, date=None, title=None):
    """
    Plots daily and cumulative profit, price change

    Parameters
    ----------
    profit : Series or one-dimensional numpy array
        PnL
    target : Series or one-dimensional numpy array
        target
    prediction : Series or one-dimensional numpy array
        model prediction
    date : index or one-dimensional numpy array, optional
        If not None, x-values of the plots
        The default is None.
    title : str, optional
        If not None, the figure title. 
        The default is None.

    Returns
    -------
    plt.fig, plt.ax

    """
    
    if date is None:
        date=np.arange(len(profit))
        
        
    assert len(profit) == len(target) == len(prediction),\
        'profit, target and prediction must have the same length'  
        
    fig, ax = plt.subplots(3, 1, figsize=(20, 15))
    if title is None:
        title = 'PnL' 
    fig.suptitle(title)


    
    # cumulative profit
    ax[0].plot(date, np.cumsum(profit), label='cumulative profit')
#    x=0
#     for year, quarter, l in len_quarters:
#         x += l
#         ax[1].axvline(x=x)
    ax[0].legend()

    # bar plots with date don't show all the bars for some reason
    date=np.arange(len(profit))
    
    # day profit
    color = ['green' if pred > 0 else 'blue' for pred in prediction]
    ax[1].bar(date, profit, color=color, label='day profit')
    # ax[1].plot(date, profit, label='day profit')
    ax[1].legend()
    
    # day profit
    ax[2].bar(date, target, label='daily price change')
    ax[2].legend()
    
#     # cumulative return
#     ax[1].plot(np.arange(len(return_day)), np.cumprod(return_day + 1) - 1, label='cumulative return')
#     ax[1].legend()
    
      
    
    plt.show()
    
    return fig, ax


###############################################################################
def plot_profit_two(profit1, target1, decision1,\
                    profit2, target2, decision2,\
                    date1=None, date2=None, title=None, ):
    """
    Similar for plot_profit(), but for two sets of profit, target and decision,
    used to compare results for two models.

    Parameters
    ----------
    See the description of the corresponding parameters in plot_profit().
    """
    
    if date1 is None:
        date1=np.arange(len(profit1))
    
    if date2 is None:
        date2=np.arange(len(profit2))
    
    assert len(profit1) == len(target1) == len(decision1),\
        'profit, target and prediction must have the same length'  
        
    assert len(profit2) == len(target2) == len(decision2),\
        'profit, target and prediction must have the same length'      
    
    fig, ax = plt.subplots(3, 1, figsize=(20, 15))
    if title is None:
        title = 'PnL for models 1 and 2' 
    fig.suptitle(title)


    
    # cumulative profit
    ax[0].plot(date1, np.cumsum(profit1), color='black', label='cumulative PnL 1')
    ax[0].plot(date2, np.cumsum(profit2), color='grey', label='cumulative PnL 2')

    ax[0].legend()

    # bar plots with date don't show all the bars for some reason
    # dat=np.arange(len(profit))
    
    # day profit
    color1 = ['green' if pred > 0 else 'blue' for pred in decision1]
    color2 = ['green' if pred > 0 else 'blue' for pred in decision2]
    ax[1].scatter(date1, profit1, color=color1, marker='^', label='PnL 1')
    ax[1].scatter(date2, profit2, color=color2, marker='v', label='PnL 2')
    ax[1].plot(date1, np.zeros(len(date1)), color='grey')
    ax[1].plot(date2, np.zeros(len(date2)), color='grey')
    # ax[1].plot(date, profit, label='day profit')
    ax[1].legend()
    
    # day profit
    ax[2].bar(np.arange(len(target1)), target1, color='black', label='Price change 1')
    # ax[2].bar(np.arange(len(target2)), target2, color='grey', label='Price change 2')
    ax[2].legend()
    
#     # cumulative return
#     ax[1].plot(np.arange(len(return_day)), np.cumprod(return_day + 1) - 1, label='cumulative return')
#     ax[1].legend()
    
      
    
    plt.show()
    
    return fig, ax



###############################################################################

def plot_reg_prediction(target, prediction, date=None, title=None, **kwargs):
    """
    Plot predictions of a regression type model. 

    Parameters
    ----------
    target : Series or one-dimensional numpy array
        target
    prediction : Series or one-dimensional numpy array
        model prediction
    date : index or one-dimensional numpy array, optional
        If not None, x-values of the plots
        The default is None.
    title : str, optional
        If not None, the figure title. 
        The default is None.

    **kwargs : additional keywords
    
    window_size : int, optional
        Size of the rolling window for moving RMSE plot.
        The default is 10.
       
    Returns
    -------
    plt.fig, plt.ax


    """
    
    if date is None:
        date=np.arange(len(target))
        
    fig, ax = plt.subplots(3, 1, figsize=(20, 15))
    if title is None:
        title = 'Regression prediction' 
    fig.suptitle(title)

    
    # target and regression predictions
    ax[0].plot(date, target, color='blue', label='target')
    ax[0].plot(date, prediction, color='green', label='prediction')
    ax[0].legend()
    
    # plotting error between prediction and target
    error = prediction - target
    ax[1].plot(date, error, color='red', label='error')
    ax[1].legend()
    
    
    # plotting error  moving average 
    
    window_size = kwargs.get('window_size', 10)    
    y = error.pow(2).rolling(window_size).mean().apply(np.sqrt)
    ax[2].plot(date, y, color='red', label=f'moving RMSE with window size {window_size}')
    ax[2].legend()
      
    
    plt.show()
    
    return fig, ax 


###############################################################################

def plot_reg_prediction_two(target1, prediction1, target2, prediction2,
                            date1=None, date2=None, title=None, **kwargs):
    """
    Similar for plot_reg_prediction(), but for two sets of target and prediction,
    used to compare results for two models.

    Parameters
    ----------
    See the description of the corresponding parameters in  plot_reg_prediction().
    """
    
    if date1 is None:
        date1=np.arange(len(target1))
        
    if date2 is None:
        date2=np.arange(len(target2))    
        
    fig, ax = plt.subplots(3, 1, figsize=(20, 15))
    if title is None:
        title = 'Regression prediction' 
    fig.suptitle(title)

    
    # target and regression predictions
    ax[0].plot(date1, target1, color='blue', label='target')
    ax[0].plot(date1, prediction1, color='red', label='prediction 1')
    ax[0].plot(date2, prediction2, color='green', label='prediction 2')
    ax[0].legend()
    
    # plotting error between prediction and target
    error1 = prediction1 - target1
    error2 = prediction2 - target2
    ax[1].plot(date1, error1, color='red', label='error 1')
    ax[1].plot(date2, error2, color='green', label='error 2')
    ax[1].legend()
    
    
    # plotting error  moving average 
    
    window_size = kwargs.get('window_size', 10)    
    y1 = error1.pow(2).rolling(window_size).mean().apply(np.sqrt)
    y2 = error2.pow(2).rolling(window_size).mean().apply(np.sqrt)
    ax[2].plot(date1, y1, color='red', label=f'moving RMSE 1 with window size {window_size}')
    ax[2].plot(date2, y2, color='green', label=f'moving RMSE 2 with window size {window_size}')
    ax[2].legend()
      
    
    plt.show()
    
    return fig, ax 

  # ttis list of  TrainTest instances to compare
  # input_set set for which models are compared: 'test', 'cross_val',...
############################################################################### 
###############################################################################    
def compare_models(ttis, input_set='test', path_prefix=None, verbose=1, **kwargs):
    """
    Compore predictions and performances of several TrainTest instances for 
    a given input_set, plots the results and save the plots. 

    Parameters
    ----------
    ttis : list-like
       list of TrainTest instances to compare
    input_set : str, optional
        Name of the set, for which performance is calculated, e.g.
        'test', 'train', 'cross_val_', etc.
        The default is 'test'.    
    path_prefix : str
        Path prefix for saving plots.
    verbose : int
        verbosity mode
        The default is 1.    
   
    **kwargs : optional keywords
   
    model_1 : int, optional
        Index in ttis of the first model 
        (some comparisons are done for two models only).
        The default is 0.
    model_2 : int, optional
        Index in ttis of the first model 
        (some comparisons are done for two models only).
        The default is 1.
    window_size : int, optional
        Window size for rolling RMSE plot.
        The default is 10.
    filename : str, optional
        If not None, file name for saving output.
        The default is None.
   
    Returns
    -------
    output_dic : dictionary
        Collection of DataFrames with various model charactreristics.
    df_pnl_time : DataFrame
       Models performance for index groups (time periods).

    """

    if path_prefix is None:
        path_prefix='//trs-fil01/trailstone/Alexander Ossipov/Projects/Plots/Temp/'
    
    path_prefix = os.path.join(path_prefix, 'Comparison/')
    m = len(ttis)
    
    for tti in ttis:
        assert input_set in tti.set_dic.keys(), 'input_set parameter is not valid'
        
    X_y_sets = [tti.set_dic[input_set] for tti in ttis]
    
    for i in range(m):
          assert X_y_sets[i] is not None, f'Create {input_set} set first for model {i + 1}'
    
    y_sets = [X_y_set['y'] for X_y_set in X_y_sets]
    
    trading = all(tti.trading for tti in ttis)  # True iff all models"trading" models, i.e. predict PnL    
    
    output_dic = {}
    
    for i in range(m):
          assert 'Decision' in y_sets[i].columns, f'Get first model performance for model {i + 1}'

    
    if trading:
        # collect performance of all models in a single DataFrame
        model_performances = [tti.model_performance_total[input_set] for tti in ttis]
        
        for i in range(m):
              assert model_performances[i] != {}, f'Get first model performance for model {i + 1}'
              assert 'PnL' in y_sets[i].columns, f'Get first model performance for model {i + 1}'
      
        df_performances = [pd.DataFrame(model_performances[i], index=[i + 1]) for i in range(m)]  # performnce of each model
        df_all_models_performance = pd.concat(df_performances)  # performances of all models in one DataFram
        df_all_models_performance.index.name = 'Model'  
        
        if verbose > 0:
            print(df_all_models_performance)
        
        output_dic['perfomance'] = df_all_models_performance
        
    
    str_title = f'{input_set=}//'  # string for html file
    
   
    
    # correlations between target, prediction and decision for each model
    corr_sets = [y_sets[i][[ttis[i].pred_col_name, ttis[i].target_col_name, 'Decision', 'PnL']].corr() for i in range(m)]\
                if trading else\
                [y_sets[i][[ttis[i].pred_col_name, ttis[i].target_col_name]].corr() for i in range(m)]  
    
    # plot correlations
    
    fig_model_corr, ax = plt.subplots(m, figsize=(8, 8 * m))
    
    model_titles = []
    for i, corr_set in enumerate(corr_sets):
        sn.heatmap(corr_set.astype(float), ax=ax[i], annot=True)
        title = 'Correlations for ' 
        
        model = ttis[i].model.__class__.__name__
        position_size_col = ttis[i].performance_parameters.get('position_size_col', None)\
                            if ttis[i].performance_parameters else None
        ind = 0 if ttis[i].prediction_type == 'regression' else 1
        target = ttis[i].label_columns[ind]
        model_name = f' {ttis[i].model_name}' if ttis[i].model_name else ''
        model_title =  f' model {i + 1}{model_name}: {model}// {target=}//  {position_size_col=}//'
        model_titles.append(model_title)
        title +=  model_title
        str_title += model_title
        ax[i].set_title(title)
   
    
    figs = {'models_correlation': fig_model_corr}
      
    
    # find common index for all models
    common_index = y_sets[0].index
    for i in range(1, m):
        common_index = common_index.intersection(y_sets[i].index)
   
    # decision correlation of different models
    cols = [f'Decision model {i + 1}' for i in range(m)]
    df_decisions = pd.DataFrame(index=common_index, columns=cols)
    for i in range(m):
        df_decisions.iloc[:, i] = y_sets[i].loc[common_index, 'Decision'].values
    
    decision_corr = df_decisions.corr()
    corr_dic = {'Decision correlation': decision_corr.astype(float)}
    
   
    output_dic['decisions'] = df_decisions
    output_dic['decision_corr'] = decision_corr
   
    # prediction correlation of different models
    cols = [f'Prediction model {i + 1}' for i in range(m)]
    df_predictions = pd.DataFrame(index=common_index, columns=cols)
    for i in range(m):
        df_predictions.iloc[:, i] = y_sets[i].loc[common_index, ttis[i].pred_col_name].values
    
    prediction_corr = df_predictions.corr()
    corr_dic['Prediction correlation'] = prediction_corr.astype(float)
    
   
    output_dic['predictions'] = df_predictions
    output_dic['prediction_corr'] =  prediction_corr
    
    
    
    if trading: 
        # PnL correlations of different model 
        cols = [f'PnL model {i + 1}' for i in range(m)]
        df_pnls = pd.DataFrame(index=common_index, columns=cols)
        for i in range(m):
            df_pnls.iloc[:, i] = y_sets[i].loc[common_index, 'PnL'].values
        
        pnl_corr = df_pnls.corr()
        
        output_dic['pnls'] = df_pnls
        output_dic['pnl_corr'] = pnl_corr
    
        corr_dic['PnL correlation'] = pnl_corr.astype(float)
        
        
    # plot correlations
    n = len(corr_dic)
    fig_decision, ax = plt.subplots(n, figsize=(8, 8 * n), squeeze=False)
    ax = ax.ravel()
    
    for i, (key, corr) in enumerate(corr_dic.items()):      
        sn.heatmap(corr, ax=ax[i], annot=True)
        ax[i].set_title(key)
    plt.show()
   
    fig_decision_corr_title = 'decision_prediction_pnl_correlation' if trading else 'decision_prediction_correlation'
    figs[fig_decision_corr_title] = fig_decision
    
    # plot PnLs for two models
    # their indices are apecified in the arguments
    # if not, they are 0 and 1
    i1 = kwargs.get('model_1', 0)
    i2 = kwargs.get('model_2', 1)
    target1 =  y_sets[i1]['y_cont']
    target2 =  y_sets[i2]['y_cont']
    date1, date2 = y_sets[i1].index, y_sets[i2].index 
    
    if trading:
    
        pnl1, pnl2 = y_sets[i1]['PnL'],  y_sets[i2]['PnL']  
        decision1, decision2 = y_sets[i1]['Decision'],  y_sets[i2]['Decision']  
        title = f'PnL for {model_titles[i1]} {model_titles[i2]}'
        
        
        fig_two_pnl, ax_two = plot_profit_two(pnl1, target1, decision1,\
                        pnl2, target2, decision2,\
                        date1, date2, title)
        
            
        figs['pnls'] = fig_two_pnl   
        # sum of PnLs
        
        total_pnl = df_pnls.sum(axis=1)
        # total_decision = df_decisions.sum(axis=1)    
        num_positions = len(total_pnl[total_pnl != 0])
        num_days = max(tti.num_days for tti in ttis)
        total_volume = sum(tti.model_performance_total[input_set]['total volume'] for tti in ttis)
        total_performance = pnl_metric(total_pnl, num_days, num_positions, total_volume) 
        
        if verbose > 0:
            print('\n\nPerformance for sum of PnLs:\n')
            [print(f'{key}: {value:.2f}', end='  ') for key, value in total_performance.items()]
            print()
        
        output_dic['total_performance'] = total_performance
    
    else:
        prediction1 =  y_sets[i1][ttis[i1].pred_col_name]
        prediction2 =  y_sets[i2][ttis[i2].pred_col_name]
        title = f'Regression predictions for {model_titles[i1]} {model_titles[i2]}'
        fig_two_reg, ax_two = plot_reg_prediction_two(target1, prediction1, target2, prediction2,
                                    date1, date2, title)
        
        figs['reg_predictions'] = fig_two_reg 
    
    # rolling RMSE for regression/accuracy for classification
    window_size = kwargs.get('window_size', 10)
    fig_error, ax = plt.subplots(figsize=(8, 4))
    for i in (i1, i2):
        if ttis[i].prediction_type == 'regression':
            error = y_sets[i]['Reg_prediction'] - y_sets[i]['y_cont']
            y = error.pow(2).rolling(window_size).mean().apply(np.sqrt)
        else:
            prob_pred = y_sets[i][ttis[i].pred_col_names]
            binary_pred = prob_to_label_multiclass(prob_pred)
            error = (binary_pred != y_sets[i]['y_binary'])
            y = error.rolling(window_size).mean()
        ax.plot(y.index, y)
    ax.legend([model_titles[i1], model_titles[i2]])
    if ttis[i1].prediction_type == 'regression':
        ax.set_title(f'Moving RMSE with window size {window_size}')
    else:
        ax.set_title(f'Moving accuracy with window size {window_size}')
    plt.show()      

    figs['moving_error'] = fig_error 
    # compare performance metrics for different time periods
    if verbose > 0:
        print('\nCompare partial performance:\n')
    for  i in (i1, i2):
        tti = ttis[i]  
        assert tti.df_performance_groups is not None, 'tti.df_performance_groups() should be called before compare_models()'
        if i == i1:
            df_pnl_time_1 = tti.df_performance_groups.add_suffix(f' {1}')            
        else:    
            df_pnl_time_2 = tti.df_performance_groups.add_suffix(f' {2}')
              
    df_pnl_time = df_pnl_time_1.join(df_pnl_time_2, how='outer')
    
    #reorder columns
    cols = []
    for col1, col2 in zip(df_pnl_time_1.columns, df_pnl_time_2.columns):
        cols.extend([col1, col2])
    
    df_pnl_time = df_pnl_time[cols]
    if verbose > 0:
        print(df_pnl_time.iloc[:,:6])  
    
    # plot df_pnl_time and add to figures to save
    title = f'Performance for {model_titles[i1]} {model_titles[i2]}'
    fig_pnl_time, ax_pnl_time = tsv.plotDFtable(df_pnl_time, strTitle=title, 
                                              equalFormat=False)
    figs[f'performance_seasons_{input_set}'] = fig_pnl_time
    
    
    # choose the target name as the one for the first model
    ind = 0 if ttis[0].prediction_type == 'regression' else 1
    target = ttis[0].label_columns[ind]
    save_path = f'{path_prefix}{Strings.slugify(target)}/'
    save_plots(figs, save_path, str_title=str_title)
    
    filename = kwargs.get('filename', None)
    if filename is not None:
        df_pnl_time.to_csv(filename)
        dump(output_dic, filename.replace('.csv', '_dic.joblib'))
  
    return output_dic, df_pnl_time

###############################################################################    
def create_error_model(tti, k=1, lag_factor=1, model='linear', comparison_file=None, plot_path_prefix=None,
                       make_prediction=True, calculate_performance=True, verbose=0):
    """
    Create models for errors for TrainTest instance ("error model"). Using such a model generate
    a new TrainTest instance with the predictions calculated as a sum of the
    original model and the error model predictions ("sum model"). 
    Compare the original model with the "sum model".
    For classifiers the error is defined  as the differnce between the probability 
    prediction and the corresponding label (0 or 1).

    Parameters
    ----------
    tti : TrainTest() object
        Original model
    k : int, optional
        The features of the error model are previous predictions errors. 
        The maximium lag is equal to k * lag_factor. For example, for k=1 only the error
        from the date (row of tti.df) = curent date - lag_factor is taken into account (one feature),
        for k=2, errors form the two previous dates (rows) = curent date - lag_factor 
        and curent date - 2 * lag_factor are taken into account (two features).
        The default is 1.
    lag_factor : int, optional
        Determines how lags are calculated, see definiton of k.
        The default is one.
    model : Regressor, optional
        Model for predicting errors of the original model.
        If model='linear', the standard linear regression is used.
        The default is 'linear'.
    comparison_file : str, optional
        If not None, file name for saving output of the comparison.
        The default is None.
    plot_path_prefix : str
        Path prefix for saving plots.
    make_prediction : boolean, optional
        If True, the "sum model" is used to make a prediction for a 
        single row of data following the last row of data in the test set
        of the original model.
        The default is True.
    calculate_performance : boolean, optional
        If True, the performances of the original model and the sum model
        are calculated and compared.
        The default is True.     
    verbose : int
        verbosity mode
        The default is 0.

    Returns
    -------
    tti_error : TrainTest() object
        "error model"
    tti_sum : TYPE
        "sum model"   
    output_dic : dictionary
        Collection of DataFrames with various model charactreristics for 
        the original and "sum models".
    df_pnl_time : DataFrame
        Models performance for index groups (time periods) for    
        the original and "sum models".

    """
    
    if model == 'linear':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
    
    assert tti.X_y_train is not None,  'Create train set first'
    assert tti.X_y_test is not None,  'Create test set first'
    
    # make prediction for the next row in df followimg after the last row in tti.X_y_test['y']
    if make_prediction:
        last_test_index = tti.X_y_test['y'].index[-1]
        ind = tti.df.index.get_loc(last_test_index)
        if len(tti.df) <= ind + 1:
            print('\nWARNING: no data available for prediction')
            make_prediction = False
        else:
            # append prediction to X_y_test
            X = tti.df[ind + 1 - (tti.input_width - 1): ind + 2]
            tti.predict(X, prediciton_set_name='prediction_next')
            tti.X_y_test['y'] = pd.concat([tti.X_y_test['y'], tti.X_y_prediction['y']])
            
    # calculate error between target and prediction for train and test sets
    error_test =  tti.X_y_test['y'][tti.target_col_name] - tti.X_y_test['y'][tti.pred_col_name]
    tti.X_y_test['y']['error'] = error_test
    error_train =  tti.X_y_train['y'][tti.target_col_name] - tti.X_y_train['y'][tti.pred_col_name]
    tti.X_y_train['y']['error'] = error_train
    
    # combine test and train errors
    error = error_test.combine_first(error_train)
    error = pd.DataFrame(error, columns=['error'])    
    # len_error = len(error)     
    
    extended_error = deepcopy(error.dropna())
    delta_index = min(extended_error.shift(1).dropna().index -  extended_error.index[:-1])
    new_error_index =  pd.Index([extended_error.index[-1] + k * delta_index for k in range(1, lag_factor + 1)])
    extended_error = extended_error.reindex(extended_error.index.append(new_error_index))
    # create error lags
    for l in range(1, k + 1):
        extended_error[f'error_lag_{l}'] = extended_error['error'].shift(l * lag_factor).fillna(method='bfill')

    extended_error['lag'] = 0
    extended_error.loc[extended_error.index[-lag_factor: ], 'lag'] = np.arange(1, lag_factor + 1)
    
    
    # create error lags
    for l in range(1, k + 1):
        error[f'error_lag_{l}'] = error['error'].shift(l * lag_factor).fillna(method='bfill')
        
    df_new = tti.df.copy()
    # df_new = error
    df_new = df_new.join(error, how='outer').fillna(method='bfill').fillna(method='ffill')
    df_new['binary error'] = binary_return(df_new['error'])
    
    validation = tti.validation
    fixed_size_training = tti.fixed_size_training
    overlap_training = tti.overlap_training 
    train_first_len = tti.train_first_len
    val_len = tti.val_len
    train_len = tti.train_len
    test_len = tti.test_len
    label_columns = ['error', 'binary error'] 
    selected_columns = [f'error_lag_{l}' for l in range(1, k + 1)]
    input_width = 1 
    num_days = 0   
    trading = False 
    scaling = False                               
    create_x_y_train = True
    
    
    # create new tti for errors
      
    tti_error = TrainTest(df_new, model, validation, train_first_len, val_len, train_len,\
                                        test_len, label_columns, selected_columns=selected_columns,\
                                        fixed_size_training=fixed_size_training, overlap_training=overlap_training,
                                        input_width=input_width, label_width=1, num_days=num_days, scaling=scaling,
                                        prediction_type='regression', trading=trading)
    
    # train and test tti_error
    tti_error.train_test(verbose=0, create_x_y_train=create_x_y_train)
    
    
    if calculate_performance:
    # seasonal performance for errors
        df_error_performance_groups = tti_error.get_performance_groups(input_set='test', verbose=verbose)
    
     
    # create new tti for "sum" of the original predictions and errors
    
    label_columns = tti.label_columns
    selected_columns = tti.selected_columns
    input_width = tti.input_width 
    num_days = tti.num_days   
    trading = tti.trading
    scaling = tti.scaling
    prediction_type = tti.prediction_type
    
    tti_sum = TrainTest(tti.df, tti.model, validation, train_first_len, val_len, train_len,\
                                        test_len, label_columns, selected_columns=selected_columns,\
                                        fixed_size_training=fixed_size_training, overlap_training=overlap_training,
                                        input_width=input_width, label_width=1, num_days=num_days, scaling=scaling,
                                        prediction_type=prediction_type, trading=trading)            
    
    # copy X_y_test and X_y_train sets 
    
    tti_sum.set_dic['test'] = deepcopy(tti.set_dic['test'])
    tti_sum.X_y_test = tti_sum.set_dic['test']
    tti_sum.set_dic['train'] = deepcopy(tti.set_dic['train'])
    tti_sum.X_y_train = tti_sum.set_dic['train']
                               
    # set predictions 
    tti_sum.X_y_test['y'][tti.pred_col_name] = tti.X_y_test['y'][tti.pred_col_name] + \
                                                 tti_error.X_y_test['y'][tti_error.pred_col_name]  
    tti_sum.X_y_train['y'][tti.pred_col_name] = tti.X_y_train['y'][tti.pred_col_name] + \
                                                 tti_error.X_y_train['y'][tti_error.pred_col_name]                                               
    
    # correct prediction for classification, as we may get results < 0 or > 1
    if tti.prediction_type == 'classification':
        x =  tti_sum.X_y_test['y'][tti.pred_col_name]
        tti_sum.X_y_test['y'][tti.pred_col_name][x < 0] = 0
        tti_sum.X_y_test['y'][tti.pred_col_name][x > 1] = 1
        
        x =  tti_sum.X_y_train['y'][tti.pred_col_name]
        tti_sum.X_y_train['y'][tti.pred_col_name][x < 0] = 0
        tti_sum.X_y_train['y'][tti.pred_col_name][x > 1] = 1
                   
    # save prediction in a separate set and remove it from X_y_test        
    if make_prediction:  
        tti.X_y_test['y'] =   tti.X_y_test['y'][:-1]     
        tti_sum.set_dic['prediction_next'] = deepcopy(tti.set_dic['prediction_next'])
        sum_pred = tti_sum.set_dic['prediction_next']['y'][[tti.pred_col_name, 'Set_id']]
        sum_pred[tti.pred_col_name] = tti_sum.X_y_test['y'][tti.pred_col_name].values[-1]
        sum_pred['Set_id'] = tti.X_y_test['y']['Set_id'].values[-1]
        tti_sum.set_dic['prediction_next']['y'] = sum_pred
        tti_sum.X_y_test['y'] =   tti_sum.X_y_test['y'][:-1]
       
         
    if calculate_performance:                
    # seasonal performance for sum
        position_size_col = tti.performance_parameters.get('position_size_col', None)\
                            if tti.performance_parameters else None
        df_sum_performance_groups = tti_sum.get_performance_groups(input_set='test', delta=tti.delta, 
                                                                   position_size_col=position_size_col,
                                                                   verbose=verbose)                 
                      
        tti_sum.model_name = 'sum_with_error_prediction' 
    
        if tti.trading: 
            tti_sum.plot_performance(path_prefix=plot_path_prefix, input_set='test')
        else:
            tti_sum.plot_prediction(path_prefix=plot_path_prefix, input_set='test')    
                     
        output_dic, df_pnl_time = compare_models([tti,tti_sum], input_set='test', 
                                                  filename=comparison_file,
                                                  path_prefix=plot_path_prefix, verbose=verbose) 
    else:
        output_dic = None
        df_pnl_time = None
        
    tti_error.extended_error = extended_error
    
    return tti_error, tti_sum, output_dic, df_pnl_time


###############################################################################    
def create_error_as_feature_model(tti, model_for_err, model_err=None, err_features=None,\
                                  sum_model=True, err_shift=0, exclude_train_set=False, 
                                  calculate_performance=True, verbose=0):
    """
    Get tti object (model 1) as input. Create 2 new tti obects as an output, 
    using the following steps: 

    1. We get an error term as a difference between predictions of model 1 and 
    target.

    2. We train a new model on original features to predict
    the error term obtained in step 2. This is model 2.

    3. We include the predicted error obtained by model 2 as an additional feature.

    4. We train another model on original features and
    the error term. This is model 3.

    5. Models 2 and 3 are used to make predictions.
    
    For classifiers the error is defined  as the differnce between the probability 
    prediction and the corresponding label (0 or 1).

    Parameters
    ----------
    tti : TrainTest object for regression
        Corresponds to the original model (model 1)
    model_for_err : regression model
        model for predicting errors (model 2)
    model_err : regression model, optional
        model predicting the same target as model 1, 
        which has an additional feature -- error,
        predicted by model 2, optional
        If None, the same regressor as in model 1 is used.
        The default is None.
    err_features : str or list, tuple, set of str, optional
        features for the model predicting errors (model 2)
        If None, the same features as for model 1 are used
    sum_model : boolean, optional
        if True, the "sum model"  for model 1 and model 2 is 
        returned instead of model 3
    err_shift : int, optional
        shifts error for prediction 
        The default is 0.
    calculate_performance :boolean, optional
        If True, the performances of model 2 amd model 3
        are calculated
        The default is True.
     verbose : int
         verbosity mode
         The default is 0.


    Returns
    -------
    tti_for_err :  TrainTest object
        Corresponds to model 2.
    tti_err : TrainTest object
        Corresponds to model 3.
    df_for_err_performance_groups : DataFrame
        performance of model 2
    df_err_performance_groups : DataFrame
        performance of model 3

    """
   

    assert tti.X_y_train is not None,  'Create train set first'
    assert tti.X_y_test is not None,  'Create test set first'
    
    # include error term      
       
    # calculate error between target and prediction for train and test sets
    error_test =  tti.X_y_test['y'][tti.target_col_name] - tti.X_y_test['y'][tti.pred_col_name]
    tti.X_y_test['y']['error'] = error_test
    error_train =  tti.X_y_train['y'][tti.target_col_name] - tti.X_y_train['y'][tti.pred_col_name]
    tti.X_y_train['y']['error'] = error_train
    
    # combine test and train errors or choose the test set only
    error = error_test if exclude_train_set else error_test.combine_first(error_train)
    
    # error = pd.DataFrame(error, columns=['error'])    
    
    df_err = tti.df.copy()
    
    df_err['dummy'] = 0  # this column plays the role of the binary target column
    
    
    df_err['Prediction_error'] = error
    
    df_err['Prediction_error'] = df_err['Prediction_error'].shift(err_shift)
    df_err.dropna(inplace=True)
    # selected_columns_err = tti.selected_columns + ['Prediction_error']
    
    
    # Model for error term
    
    # print('\n******* Test model for error term *******\n')
    
    validation = False
    
    fixed_size_training = True 
    overlap_training = True 
    
    
    # init TrainTest innstance (tti)
    
    train_len = tti.train_len - (tti.input_width + err_shift)
    test_len = tti.test_len
    train_first_len = 1
    val_len = 0
    
    input_window = 1
    num_days = 0 
   
    selected_columns = err_features if err_features else tti.selected_columns
    label_columns_err = ['Prediction_error', 'dummy']
    
    tti_for_err = TrainTest(df_err, model_for_err, validation, train_first_len, val_len, train_len,\
                    test_len, label_columns_err, selected_columns=selected_columns,\
                    fixed_size_training=fixed_size_training, overlap_training=overlap_training,
                    input_width=input_window, label_width=1, num_days=num_days, scaling=False,
                    prediction_type='regression', trading=False)
    
    # train and test tti
    
    tti_for_err.train_test(verbose=0)       
    
    
    
   
    if calculate_performance:
    # seasonal performance for errors
        df_for_err_performance_groups = tti_for_err.get_performance_groups(input_set='test', verbose=verbose)
    else:
        df_for_err_performance_groups = None    
    
    # y_set = tti_for_err.X_y_test['y']    
    # plot_reg_prediction(y_set['y_cont'], y_set['Reg_prediction'], tti_for_err.index.test_total_y) 
    
    if sum_model:
        # create new tti for "sum" of the original predictions and errors
        
        label_columns = tti.label_columns
        selected_columns = tti.selected_columns
        input_width = tti.input_width 
        num_days = tti.num_days   
        trading = tti.trading
        scaling = tti.scaling
        prediction_type = tti.prediction_type
        
        tti_sum = TrainTest(tti.df, tti.model, validation, train_first_len, val_len, train_len,\
                                            test_len, label_columns, selected_columns=selected_columns,\
                                            fixed_size_training=fixed_size_training, overlap_training=overlap_training,
                                            input_width=input_width, label_width=1, num_days=num_days, scaling=scaling,
                                            prediction_type=prediction_type, trading=trading)            
        
        # copy X_y_test and X_y_train sets 
        
        tti_sum.set_dic['test'] = deepcopy(tti.set_dic['test'])
        tti_sum.X_y_test = tti_sum.set_dic['test']
        tti_sum.set_dic['train'] = deepcopy(tti.set_dic['train'])
        tti_sum.X_y_train = tti_sum.set_dic['train']
                                   
        # set predictions 
        tti_sum.X_y_test['y'][tti.pred_col_name] = tti.X_y_test['y'][tti.pred_col_name] + \
                                                     tti_for_err.X_y_test['y'][tti_for_err.pred_col_name]  
        tti_sum.X_y_train['y'][tti.pred_col_name] = tti.X_y_train['y'][tti.pred_col_name] + \
                                                     tti_for_err.X_y_train['y'][tti_for_err.pred_col_name]                                               
        
        # correct prediction for classification, as we may get results < 0 or > 1
        if tti.prediction_type == 'classification':
            x =  tti_sum.X_y_test['y'][tti.pred_col_name]
            tti_sum.X_y_test['y'][tti.pred_col_name][x < 0] = 0
            tti_sum.X_y_test['y'][tti.pred_col_name][x > 1] = 1
            
            x =  tti_sum.X_y_train['y'][tti.pred_col_name]
            tti_sum.X_y_train['y'][tti.pred_col_name][x < 0] = 0
            tti_sum.X_y_train['y'][tti.pred_col_name][x > 1] = 1
    
        if calculate_performance:
        # seasonal performance for errors
            df_sum_performance_groups = tti_sum.get_performance_groups(input_set='test', verbose=verbose)
        else:
            df_sum_performance_groups = None 
            
        return tti_for_err, tti_sum, df_for_err_performance_groups, df_sum_performance_groups    
            
    
    X = tti_for_err.df[selected_columns]
    
    X = tti_for_err.convert_window_input(X)
   
    
    # if tti_for_err.scaling:
    #     ##############################################
    #     # Standard scaling
    #     scaler = tti.scalers[ind]
    #     X = scaler.transform(X)
     
    # predicted_error = tti_for_err.fitted_models[-1].predict(X)
    # df_err = df_err[-len(predicted_error):]     
    # df_err['Model_predicted_error'] = predicted_error
    predicted_error = tti_for_err.X_y_test['y'][tti_for_err.pred_col_name] 
    df_err.loc[predicted_error.index, 'Model_predicted_error'] = predicted_error.values
    df_err.dropna(inplace=True)
    selected_columns_err = tti.selected_columns + ['Model_predicted_error']
    if not model_err:
        model_err = tti.fitted_models[-1]

    #  model with error term as a feature       
       
    # print('\n******* Test with error term *******\n')
    # init TrainTest innstance (tti)
    
    label_columns = tti.label_columns
    
    input_window = tti.input_width
    num_days = tti.num_days
    fixed_size_training = tti.fixed_size_training
    overlap_training = tti.overlap_training
    scaling = tti.scaling    
    trading = tti.trading

   
    tti_err = TrainTest(df_err, model_err, validation, train_first_len, val_len, train_len,\
                    test_len, label_columns, selected_columns=selected_columns_err,\
                    fixed_size_training=fixed_size_training, overlap_training=overlap_training,
                    input_width=input_window, label_width=1, num_days=num_days, scaling=scaling,
                    prediction_type='regression', trading=trading)
    
    # train and test tti
    
    tti_err.train_test(verbose=0)       
    
    
    if calculate_performance:
    # seasonal performance for errors
        df_err_performance_groups = tti_err.get_performance_groups(input_set='test', verbose=verbose)
    else:
        df_err_performance_groups = None    
        
    # plot_reg_prediction(y_set['y_cont'], y_set['Reg_prediction'], tti_err.index.test_total_y)     
    # error = y_set['y_cont'] - y_set['Reg_prediction']    
    # error.plot()
    
    return tti_for_err, tti_err, df_for_err_performance_groups, df_err_performance_groups

############################################################################## 
############################################################################## 
def target_feature_bin_stat(df, target_col, bins=5, quantiles=True, q_min=0.1, q_max=0.9):
    """
    Bin target into bins then calculate mean
    features in df w.r.t. that bins.
    Normalise the result by subtracting mean(feature) and dividing by
    std(feature).
    Additionaly calculate fraction of points between quatiles q_min and
    q_max for a given PnL bin relative to the total fraction of points
    between quatiles q_min and q_max.
    
    Return a DataFrame with the corresponding statistics.
   
    Parameters
    ----------
    target_col : str
        Name of the target column in df
    bins : int or array
        (Number of) bins. See pd.cut for more info.
        The default value is 5.
    quantiles: boolean
        If True use pd.qcut for quantile-based discretization,
        otherwise use pd.cut to bin values. 
        The default is True.
    q_min: float
        Min quantile
    q_max: float
         Max quantile    
        


    Returns
    -------
    df_stat_mean : DataFrame
        DataFrame with statistics of each feature w.r.t. target bins.

    """
    
    def count_points(a):
        x1, x2 = np.quantile(a, q_min), np.quantile(a, q_max)          
        return len(a[(x1 <= a) & (a <= x2)])

    def get_quantiles(a):
        return (np.quantile(a, q_min), np.quantile(a, q_max))


   
    df_copy = df.copy()
    col = target_col

    num_values =  len(df[col].unique())
    if num_values == 1:
        print(f'WARNING: Column {col} contains a single value only and will be ignored')
        return pd.DataFrame()

    if isinstance(bins, int) and num_values < bins:
        print(f'WARNING: The number of bins will be reduced to {num_values}, which is the number of unique values for column {col}.')
        df_copy[col + ' bins'] = df[col]
    else:     
        df_copy[col + ' bins'] = pd.qcut(df[col], bins, duplicates='drop') if quantiles else pd.cut(df[col], bins=bins)



    df_stat = df_copy.groupby(col + ' bins').agg(('mean', get_quantiles, count_points))

    df_stat.drop(col, axis=1, level=0, inplace=True)
    df_stat_mean = df_stat.xs('mean', level=1, axis=1)
    df_stat_quantiles = df_stat.xs('get_quantiles', level=1, axis=1).T
    df_stat_counts = df_stat.xs('count_points', level=1, axis=1).T


    df_mean = df.mean()
    df_std = df.std()

    for col in df_stat_mean:
       df_stat_mean[col] = (df_stat_mean[col] - df_mean[col]) / df_std[col]

    df_stat_mean = df_stat_mean.T
    df_stat_mean = df_stat_mean.sort_values(df_stat_mean.columns[0])
    df_stat_mean.dropna(inplace=True)

    df_stat_mean.columns = [str(col) for col in df_stat_mean.columns]

    for col in df_stat_quantiles.index:
        for ind, pnl_int in enumerate(df_stat_quantiles.columns):
            x1, x2  = df_stat_quantiles.loc[col, pnl_int]
            count_total = len(df_copy[(df_copy[col] >= x1) & (df_copy[col] <= x2)])
            f = df_stat_counts.loc[col, pnl_int] / count_total
            df_stat_mean.loc[col, str(pnl_int) + '_frac'] = f
            
    df_stat_mean.dropna(inplace=True)
    
    return df_stat_mean



###############################################################################
###############################################################################
# auxilary function to choose correct column name for num_days==1 and num_days!=1

def num_days_to_str(num_days):
    if num_days == 1:
        return ''
    return f'_{num_days}_days'


# return label columns for target_sym_quarter

def get_label_columns(target_sym_quarter, num_days):
    str_num_days = num_days_to_str(num_days)
    return [target_sym_quarter + ' return' + str_num_days, target_sym_quarter + ' binary return' + str_num_days] 


###############################################################################

def binary_return(x): # create classification label from price change 
    return (x > 0).astype(int)

###############################################################################
# create multiclass classification label from continuous target
def multiclass_return(x, class_boundaries, class_labels=None): 
    """
    

    Parameters
    ----------
    x :  one-dimensional array_like
        continuous target
    class_boundaries : one-dimensional or two-dimensional array_like
        boundaries determining classes
        If one-dimensional, len(class_boundaries) = number of classes - 1
        and the boundaries are the same for all x[i].
        and the boundaries might be different for each x[i].
    class_labels : one-dimensional array_like, optional
        Labels for each class. len(class_labels) =  len(class_boundaries) + 1
        If None, class_labels = np.arange(number of classes)
        The default is None.

    Returns
    -------
    y : one-dimensional np.array
        Multiclass labels. len(y) = len(x)

    """
    class_boundaries_ = np.array(class_boundaries)    
    m = class_boundaries_.shape[-1] + 1     
    if not class_labels:
        class_labels = np.arange(m)
        
    assert len(class_labels) == m, 'The number of class labels is not equal to the number of class boundaries + 1.'
    
    
    x_ = np.array(x)
    assert not np.isnan(x_).any(), 'x contains NaNs'
    y = np.zeros(len(x_), dtype=class_labels[0].dtype)
    if len(class_boundaries_.shape) == 1:
        class_boundaries_ = np.ones((len(x_), m - 1)) *  class_boundaries_
    
    y[x_ <= class_boundaries_[:, 0]] = class_labels[0]
    y[x_ > class_boundaries_[:, -1]] = class_labels[-1]
    for i in range(1, m - 1):
        y[(x_ > class_boundaries_[:, i - 1]) & (x_ <= class_boundaries_[:, i])] = class_labels[i]
        
    return y  


    
# In[49]:

    
def create_target_from_return(df):
    df_copy= df.copy()
    for col in df.columns:
        if ('return' in col) and ('binary' not in col):
            df_copy[col.replace('return', 'target cont')] = df[col].shift(-1)
            df_copy[col.replace('return', 'target')] = binary_return(df[col].shift(-1))
           
    return df_copy[:-1]


def create_binary_return(df, num_days=1):
    df_copy= df.copy()
    
    num_days_str =  num_days_to_str(num_days)
    
    
    if num_days == 1:  # return data is already available for the previous day
        for col in df.columns:
            if 'return' in col:
                new_col = col.replace('return', f'return{num_days_str}')
                df_copy.rename(columns={col:new_col}, inplace=True)
                df_copy[new_col.replace('return', 'binary return')] = binary_return(df[col])
               
        return df_copy
    

    for col in df.columns:
        if 'return' in col:
            new_col = col.replace('return', f'return{num_days_str}')
            # we raplace nan with 0, assuming that we have them for holidays
            # df_copy[new_col] = df[col].fillna(0).rolling(num_days).sum() 
            df_copy[new_col] = df[col].rolling(num_days).sum() 
            df_copy[new_col.replace('return', 'binary return')] = binary_return(df_copy[new_col])
        
    return df_copy 
 

def make_binary_cols(df, cols):
    original_cols = df.columns
    for col in cols:
        if 'change' in col:
            df[col.replace('change', 'binary change')] = binary_return(df[col])
        elif 'return' in col:
            df[col.replace('return', 'binary return')] = binary_return(df[col])  
        else:
            df[col + ' binary'] = binary_return(df[col])  
    
    new_cols =  [col for col in df.columns if col not in original_cols]        
                    
    return df, new_cols
   
##############################################################################
# Convert predicted probabilities to the label prediction given the threshold. 




def prob_to_label(pred_prob, th_min, th_max):
    return (pred_prob >= th_max).astype(int) - (pred_prob < th_min).astype(int)

def prob_to_label_multiclass(pred_prob):
    pred_prob = np.array(pred_prob)
    return np.argmax(pred_prob, axis=1)


def print_accuracy(prediction, y):
    print('Mean y:', y.mean(), ' mean prediction:', prediction.mean())
    print('Accuracy:', accuracy_score(y, prediction))


##############################################################################
##############################################################################

##############################################################################
def max_drawdown(y, ret_ind=False):
    
    if len(y) < 2:
        if ret_ind:
             return 0, None, None
        return 0
    
    if isinstance(y, pd.Series):
        x = y.values
    else:
        x = y
    
    # to treat correctly a possible max drawdown occuring at ind=0, one
    # needs to append 0 profit as an initial profit (first element of x)
    x = np.insert(x, 0, 0)
        
    i_end = np.argmax(np.maximum.accumulate(x) - x) # end of the period
    if i_end == 0:
         if ret_ind:
             return 0, None, None
         return 0
    
    i_start = np.argmax(x[:i_end]) # start of period   
    
    # max_dd = (x[i_end] / x[i_start] - 1) * 100  # relative drawdown in %
    max_dd = x[i_end] - x[i_start]
    
    if ret_ind:
        # subtract 1 from the indices, as we insert 0 as the first element before
        return max_dd, i_start - 1, i_end - 1  
    return max_dd

##############################################################################
def pnl_metric(profit, num_days, num_positions, total_volume):



 # annualized Sharpe ratio  
    # nonzero_return_day  = return_day[return_day != 0] #  take into account the only days when we trade
    # (chances that we trade and get exactly zero return are negligible)
    #sharpe = np.sqrt(252) * nonzero_return_day.mean() / nonzero_return_day.std() # sqrt(252) comes from converting daily
    # mean and std to annual mean and std
    # This approach might be not completely correct and we should take into account all trades including zero ones, if we want 
    # to convert the daily ratio to the annual one. Indeed, say, we trade once per month only, then it is not correct 
    # to multiply the result by sqrt(252), instead we should multiply by sqrt(12). In Sortino ratio zero trades are not taken
    # into account anyway.
    
#     # annualized Sharpe ratio
#     nonzero_return_day  = return_day[return_day != 0] #  take into account the only days when we trade
#     # (chances that we trade and get exactly zero return are negligible)
#     trades_year = 252 * len(nonzero_return_day) / len(return_day)  # number of non zero trades per year
#     sharpe = np.sqrt(trades_year) * nonzero_return_day.mean() / nonzero_return_day.std()
    
    # annualized Sharpe ratio based on "normal returns" = profit
    nonzero_profit  = profit[profit != 0] #  take into account the only days when we trade
    # (chances that we trade and get exactly zero return are negligible)
    trades_year = 252 * len(nonzero_profit) / (len(profit) * num_days) \
                  if len(nonzero_profit) != 0 else 0 # number of non zero trades per year
    sharpe = np.sqrt(trades_year) * nonzero_profit.mean() / nonzero_profit.std() \
             if (nonzero_profit.mean() != 0 and trades_year != 0) else 0
    
#     # annualized Sortino ratio  
#     # sortino = np.sqrt(252) * nonzero_return_day.mean() / nonzero_return_day[nonzero_return_day < 0].std()
#     # the above formula is incorrect, as we shuold devide by the number of all (non--zero) returns rather than only
#     # be the number of negative returns
    
#     negative_returns = np.array([r if r < 0 else 0 for r in return_day])
#     sortino = np.sqrt(252) * return_day.mean() / negative_returns.std()
    
    negative_profit = np.array([p if p < 0 else 0 for p in profit])
    sortino = np.sqrt(252 / num_days) * profit.mean() / negative_profit.std() \
             if profit.mean() != 0 else 0
    
    # mean_profit = profit.sum() / num_positions
    mean_profit = profit.sum() / total_volume  \
             if profit.sum() != 0 else 0
    
    max_dd = max_drawdown(profit.cumsum())  # maximum drawdown
    
    total_profit = profit.sum()  # total PnL
    
    relative_dd = (max_dd / total_profit) * 100  if max_dd != 0 else 0 # relative drawdown in %
    
    pnl_metric_dic = {'mean_PnL': mean_profit, 'total PnL': profit.sum(), 'total volume': total_volume,\
                      'num_positions': num_positions, \
                      'sharpe': sharpe, 'sortino': sortino, 'max_drawdown': max_dd, \
                      'drawdown_to_pnl': relative_dd}    
    
    return pnl_metric_dic

###############################################################################

def day_profit(target_cont, prediction, overlap_trade=False, num_days=1, position_size=1, confluent_pos=True, verbose=2):
    
    assert len(target_cont) == len(prediction),\
        'target and prediction must have the same length'  
    
    if not overlap_trade:

             
        for i in range(num_days):
            target_cont_slice = target_cont[i::num_days]
            prediction_slice = prediction[i::num_days]
            if not isinstance(position_size, (int, float)): 
                position_size_slice = position_size[i::num_days]
            else:
                position_size_slice = position_size
            
            if len(target_cont_slice) == 0:
                print('WARNING: len(target_cont_slice) = 0')
                continue
            model_performance = day_profit_single_position(target_cont_slice, prediction_slice, 
                                                           num_days, position_size_slice, confluent_pos, verbose)
            
            model_performance.pop('PnL')
            if i == 0:
                slice_performances = {key: np.array([]) for key in model_performance}
            for key, value in model_performance.items():
                slice_performances[key] = np.append(slice_performances[key], model_performance[key])

    # replace possible inf values for sortino by max
    if 'sortino' in slice_performances:
        sortino_list = slice_performances['sortino']
        if len(sortino_list[np.where(sortino_list != np.inf)]) > 0: #  if all values = inf, keep them
            sortino_list[np.where(sortino_list == np.inf)] = sortino_list[np.where(sortino_list != np.inf)].max()
        slice_performances['sortino'] = sortino_list    

    # save mean values for all slices as "total" model performance
    for key in model_performance:
        model_performance[key] = slice_performances[key].mean()
    
    if 'num_positions' in model_performance:
        model_performance['num_positions'] = int(model_performance['num_positions'])
    
    
    profit = target_cont * prediction * position_size # contains profit for all slices   
    model_performance['PnL'] = profit
    
    if (num_days !=1) and (verbose > 1):
        print('\nAverage results:')
        print(f"""Total PnL is {profit.sum() / num_days:.2f}, the annualized Sharpe ratio
              is { model_performance['sharpe']:.2f}, Sortino ratio is {model_performance['sortino']:.2f}""")
        print(f"""Number of opened positions: {model_performance['num_positions'] } in {int(len(prediction) / num_days)} days,\
              total volume: {model_performance['total volume']:.2f},  mean PnL: {model_performance['mean_PnL'] :.2f}""")    
    
    return model_performance

###############################################################################

def day_profit_single_position(target_cont, prediction, num_days=1, position_size=1, confluent_pos=True, verbose=2):
    
    # prediction = decision here, i.e. +1, -1 or 0
    
    # profit = np.zeros(len(prediction)) # daily profit = normal return
    # return_day = np.zeros_like(profit) # daily return = log return
    
    assert isinstance(target_cont[0], np.floating) or isinstance(target_cont[0], float), 'Input array for target is not float'
    
    target_cont_ = np.array(target_cont)
    prediction_ = np.array(prediction)
    
    volume = prediction_ * np.array(position_size)  # traded volume with sing + for byy and - for sell
    
    # As we close each position next day, then the return for a given position = (model prediction) * (price change) * (position_size)
    profit = target_cont_ * volume
    
    # In reality we don't want to close a position everyday. For exapmle, if we opened a long position 
    # and a model predict that the price goes up, we will keep the position. Therefore if we want to 
    # calculate the actual number of open positions we need to count repeating predictions, such as
    # 1,1,1,.. as a single position. Hence we need to count the absolute values of the prediction change. This
    # gives 2 times the numebr of opened positions, as our prediction should change every  time when we open 
    # and close the position
    
    if confluent_pos:
        num_positions = int(np.abs((np.append(prediction_, 0) - np.insert(prediction_, 0, 0))).sum() // 2)
        total_volume = np.abs((np.append(volume, 0) - np.insert(volume, 0, 0))).sum() / 2
        
    
    # If we can open a new position at the same time when we close the previous one, we should treat them 
    # separately.
    else:
        num_positions = len(prediction_[prediction_ != 0])
        total_volume = np.abs(volume).sum()
        
    # Traded volume can be calculated in the same way as the number of positions
    
    
    model_performance = {'PnL': profit}
    model_performance.update(pnl_metric(profit, num_days, num_positions, total_volume))
    
   
    sharpe = model_performance['sharpe']
    sortino = model_performance['sortino']
    mean_pnl = model_performance['mean_PnL']
    max_dd =  model_performance['max_drawdown']
    relative_dd = model_performance['drawdown_to_pnl']
    
    
   

    if verbose > 0:
        print(f'Total PnL is {profit.sum():.2f}, the annualized Sharpe ratio is {sharpe:.2f}, Sortino ratio is {sortino:.2f}')
        print(f'Number of opened positions: {num_positions} in {len(prediction)} days, total volume: {total_volume: .2f},  mean PnL: {mean_pnl:.2f}')    
        print(f'max drawdown: {max_dd:.2f}, drawdown to total PnL: {relative_dd:.1f}%')
  
    return model_performance


###############################################################################


def day_profit_offset(target_cont, prediction, date, quarter, offset, max_offset, overlap_trade=False, num_days=1):
    
    first_month = (quarter- 1) * 3 + 1  # first month of the quarter
    allowed_months = [i for i in range(1,13) if offset <= ((first_month - i) % 12) <= max_offset]
    mask = date.dt.month.isin(allowed_months)     
    profit, mean_profit, num_positions, sharpe, sortino = day_profit(target_cont[mask], prediction[mask], overlap_trade=overlap_trade, num_days=num_days)
    
    return profit, mean_profit, num_positions, sharpe, sortino
    
###############################################################################
# load TRainTest object from file
def load_tti(filename):
    
    if (filename is not None) and os.path.exists(filename):  # load tti from file if it exists
        # weather_sims_range = pd.read_pickle(filename)
        print(f'Loading TrainTest object from file: {filename}')
        tti = load(filename)
    
        return  tti


###############################################################################
# calculate various metrics of regression prediction

def regression_metrics(target, reg_prediction, decision, num_days=1):
    
    
    assert len(target) == len(reg_prediction) == len(decision),\
        'target, reg_prediction and decision must have the same length'  
        
    target = np.array(target).reshape(-1, 1)    
    reg_prediction = np.array(reg_prediction).reshape(-1, 1)    
    decision = np.array(decision).reshape(-1, 1) 
    
    reg_metrics = {}
    
    # RMSE
    reg_metrics['RMSE'] = mean_squared_error(target, reg_prediction, squared=False)
    
    # R^2
    reg_metrics['R^2'] = r2_score(target, reg_prediction)
    
    # prediction error autocorrelation
    error = (target - reg_prediction).astype(float)
    
    # if not isinstance(error, pd.Series):
    error = pd.Series(error.flatten())
    
    shift = max(num_days, 1)
    reg_metrics['error_autocorr'] = error.shift(shift).corr(error)
    
    reg_metrics['mean_error'] = error.mean()
    
    return reg_metrics
    
###############################################################################
# calculate various metrics of classification prediction

def classification_metrics(target, prediction_prob, decision, suppress_warnings=False, class_labels=None):
    
    assert len(target) == len(prediction_prob) == len(decision),\
        'target, prediction_prob and decision must have the same length'  
        
    prediction_prob_ = np.array(prediction_prob)    
    m = prediction_prob_.shape[-1]  # number of classes    
    if not class_labels:
        class_labels = np.arange(m)  
    class_labels = np.array(class_labels)    
    
    clf_metrics = {}
    
    # decision accuracy
    
    # select only values for which decision != 0
    decision_ = decision[decision != 0].astype(int)
    target_ = target[decision != 0].astype(int)
    
    pred = class_labels[prediction_prob_.argmax(axis=1)]
    pred_ = pred[decision != 0]
    prediction_prob_ = prediction_prob_[decision != 0]
    
    
    clf_metrics['decision_accuracy'] = accuracy_score(target_, pred_) \
                                       if len(decision_) != 0 else 0                                 
    
    # AUC
    try:
        if m == 2:
            clf_metrics['decision_AUC'] = roc_auc_score(target_, pred_)  \
                                               if len(decision_) != 0 else 0
        else:
            clf_metrics['decision_AUC'] = roc_auc_score(target_,  prediction_prob_, multi_class='ovr')  \
                                               if len(decision_) != 0 else 0                                          
    except ValueError as e:
        if not suppress_warnings:
            print(f'\nWARNING: {e}\n')   
            # print(target_)
            # print(pred_)
        
    # Precision
    try:
        if m == 2:
            clf_metrics['Precision'] = precision_score(target_, pred_)  \
                                               if len(decision_) != 0 else 0   
        else:
            clf_metrics['Precision'] = precision_score(target_, pred_, average='micro')  \
                                               if len(decision_) != 0 else 0                                        
    except ValueError as e:
        if not suppress_warnings:
            print(f'\nWARNING: {e}\n')   
            # print(target_)
            # print(pred_)
        
    # Positive fraction
    clf_metrics['Positive_fraction'] = len(target[target == 1]) / len(target)
   
    
    return clf_metrics
        


##############################################################################
##############################################################################
def X_y_from_ds(ds):    
    X_set, y_set  = [], []    
    data_list = list_from_dataset(ds)

    for X, y in data_list:  
        X_set.append(X.reshape(1,-1))
        y_set.append(y.reshape(1,-1))

    X_set = np.vstack(X_set)
    y_set = np.vstack(y_set)
    return X_set, y_set[:,1], y_set[:,0]
    
def X_y_from_wgen(w):
    train, test = w.train, w.test
    
    X_train, y_train, y_train_cont = X_y_from_ds(train)
    X_test, y_test, y_test_cont = X_y_from_ds(test)
    
    return X_train, y_train, y_train_cont, X_test, y_test, y_test_cont
    


##############################################################################



def train_validation_test(df, model, validation, train_first_len, val_len, train_len, test_len, 
                      label_columns, selected_columns=None, fixed_size_training=False, overlap_training=True, 
                      delta=0, input_width=1, label_width=1, overlap_trade=False, num_days=1, verbose=2,
                      return_performance=False, return_index=False, scaling=False, position_size_col=None):
    
    
    if selected_columns is None:
        selected_columns = []
    
    if not isinstance(selected_columns, (list, tuple, set)):
        selected_columns = [selected_columns]  
    
    # generator for train/test sets
    # print('before split')
    split_gen = train_test_split(df, train_first_len, val_len, train_len, test_len, 
                      label_columns, fixed_size_training, overlap_training,  
                      input_width=input_width, label_width=label_width, shift=num_days, verbose=verbose)
    
    # print('after split')
          
    if not validation:
          _ = next(split_gen)

    y_test_total = np.array([])   
    pred_test_total = np.array([])
    binary_pred_test_total = np.array([])
    y_test_cont_total = np.array([])   
    position_size_test_total = np.array([]) if position_size_col is not None else 1
    
    train_test_index = {'train':[], 'val':[], 'test':[]}
    
    # train_dates = pd.Series([])
    # test_dates = pd.Series([])

    for w in split_gen:
    
        X_train, y_train, y_train_cont, X_test, y_test, y_test_cont = X_y_from_wgen(w)
       
    #     y_train, y_train_cont = y_train.values.flatten(), y_train_cont.values.flatten()
    #     y_test, y_test_cont = y_test.values.flatten(), y_test_cont.values.flatten()

        # print(X_train.shape, y_train.shape)
        if scaling:
            ##############################################
            # Standard scaling
    
    
            # use X_train to scale X_train and X_test
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # convert numpy arrays to DataFrames    
        X_train = pd.DataFrame(X_train,  columns=df.columns)
        # X_valid = scaler.fit_transform(X_valid)
        X_test = pd.DataFrame(X_test, columns=df.columns)

        ##############################################
        # Extract positions size from df, if required
        
        position_size_train, position_size_test = 1, 1
       
        if position_size_col is not None:
            position_size_train = X_train[position_size_col]
            position_size_test = X_test[position_size_col]
            print('Position size: ',  position_size_test[1:5])    
                
        
        ##############################################
        # Exclude some columns, if required
        # this is the only way not to include features used to generate
        # labels in the train/test sets
        
        if selected_columns:
            X_train = X_train[selected_columns]
            X_test = X_test[selected_columns]
           
        
        
        
        if verbose > 1:
           print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        ##############################################
        # Model train
        if verbose > 1:
            print('\n Train \n')

        # if 'date' in X_train.columns:
        #     train_dates = pd.concat([train_dates, X_train['date']])
        #     X_train.drop(columns=['date'], inplace=True)

        model.fit(X_train, y_train) 



        binary_prediction = model.predict(X_train)
        prediction_proba = model.predict_proba(X_train)
        if verbose > 1:
            print_accuracy(binary_prediction, y_train)

        # delta = 0.15
        th_min, th_max = 0.5 - delta, 0.5 + delta 
        prediction = prob_to_label(prediction_proba[:,1], th_min, th_max)
        model_performance = day_profit(y_train_cont, prediction, 
                                        overlap_trade=overlap_trade, num_days=num_days, 
                                        position_size=position_size_train, verbose=verbose)
        # profit_train, mean_profit, num_positions, sharpe, sortino  =  day_profit_single_position(y_train_cont, prediction)
       
        # print(model_performance)
        profit_train, mean_profit, num_positions, sharpe, sortino = model_performance['PnL'], \
                                                                    model_performance['mean_PnL'],\
                                                                    model_performance['num_positions'],\
                                                                    model_performance['sharpe'],\
                                                                    model_performance['sortino']    
        
        ##############################################
        # Model test
        if verbose > 1:
            print('\n Test \n')
        
        # if 'date' in X_test.columns:
        #     test_dates = pd.concat([test_dates, X_test['date']])
        #     X_test.drop(columns=['date'], inplace=True)


        binary_pred_test = model.predict(X_test)
        if verbose > 1:
            print_accuracy(binary_pred_test, y_test)

        pred_proba_test = model.predict_proba(X_test)
        pred_test = prob_to_label(pred_proba_test[:,1], th_min, th_max)
        model_performance = day_profit(y_test_cont, pred_test, 
                                        overlap_trade=overlap_trade, num_days=num_days,
                                        position_size=position_size_test, verbose=verbose)
        # profit, mean_profit, num_positions, sharpe, sortino  =  day_profit_single_position(y_test_cont, pred_test)
        profit_test, mean_profit, num_positions, sharpe, sortino = model_performance['PnL'], \
                                                                    model_performance['mean_PnL'],\
                                                                    model_performance['num_positions'],\
                                                                    model_performance['sharpe'],\
                                                                    model_performance['sortino']  
                                                                           


        ##############################################
        # Concatenate results with the previous ones
        
        
        

        y_test_total = np.concatenate((y_test_total, y_test))
        pred_test_total = np.concatenate((pred_test_total, pred_test))
        binary_pred_test_total = np.concatenate((binary_pred_test_total, binary_pred_test))
        y_test_cont_total = np.concatenate((y_test_cont_total, y_test_cont))
        if position_size_col is not None:
             position_size_test_total = np.concatenate((position_size_test_total,  position_size_test))
            
        if return_index:
            train_test_index['train'].append((w.train_first_ind, w.train_last_ind))
            train_test_index['val'].append((w.val_first_ind, w.val_last_ind))
            train_test_index['test'].append((w.test_first_ind, w.test_last_ind))
        if validation: 
            break
    
            
        
    if verbose > 0:    
        print('\nTOTAL\n') 

    if verbose > 0: 
        print_accuracy(binary_pred_test_total, y_test_total)
    model_performance  =  day_profit(y_test_cont_total, pred_test_total,
                                    overlap_trade=overlap_trade, num_days=num_days,
                                    position_size=position_size_test_total, verbose=verbose)

    
    profit, mean_profit, num_positions, sharpe, sortino = model_performance['PnL'], \
                                                                model_performance['mean_PnL'],\
                                                                model_performance['num_positions'],\
                                                                model_performance['sharpe'],\
                                                                model_performance['sortino']  
                                                                       

    # profit, mean_profit, num_positions, sharpe, sortino  =  day_profit_single_position(y_test_cont_total, pred_test_total)
    
    # if return_dates:
    #     return profit, y_test_cont_total, pred_test_total, train_dates, test_dates
    
    # convert train_test_index
    if return_index:
        for X_set, index_list in train_test_index.items():
            first_ind, last_ind = index_list[0]
            index = None
            if first_ind != -1:
                index = df[first_ind: last_ind].index
            if len(index_list) > 1:
                for first_ind, last_ind  in index_list[1:]:
                    if first_ind != -1:
                        index = index.union(df[first_ind: last_ind].index)
            train_test_index[X_set] = index    
 
    
    if return_performance:
        if return_index:
            return profit, y_test_cont_total, pred_test_total, train_test_index, mean_profit, num_positions, sharpe, sortino    
        return profit, y_test_cont_total, pred_test_total, mean_profit, num_positions, sharpe, sortino   
    if return_index:
        return profit, y_test_cont_total, pred_test_total, train_test_index
    return profit, y_test_cont_total, pred_test_total
    


###############################################################################
# save train and test results for model performance
def save_model_results(filename, index=None, **kwargs):

    if filename is None:
                print('Warning: No file name is specified, the results won\'t be saved.')
                return
    
    if index is None:
        index = [pd.to_datetime('today').date()]        
    
    df = pd.DataFrame(kwargs, index=index)
    
    file_exist = os.path.exists(filename)
    if file_exist:
        try:
            df_old = pd.read_csv(filename, index_col=0)
            can_append = (df_old.columns  == df.columns).all()
        except:
            can_append = False
        if can_append:  # file exists and have the same columns
            try:
                df.to_csv(filename, mode='a', header=False)
            except:
                can_append = False
        if not can_append:  # file exists, but we can't append
            today = pd.to_datetime('today').strftime('%Y-%m-%d_%H-%M-%S')
            try:
                os.rename(filename, filename.replace('.csv', f'_{today}.csv'))  # rename the existing file
                df.to_csv(filename, mode='w', header=True)
                os.chmod(filename, 0o777)
            except:
                print(f'WARNING: file {filename} already exists and can\'t be modified.')
          
    else:  # file doesn't exist
        df.to_csv(filename, mode='w', header=True)
        # print('Changing file permission')
        os.chmod(filename, 0o777)
###############################################################################
# save plot to file
# figs -  dictionary {fig_name: fig}
def save_plots(figs, save_path, str_title=None, file_prefix=None):
  

   if not os.path.exists(save_path):
       os.makedirs(save_path)
       
   if not os.path.exists(save_path + 'latest.html'):
       shutil.copyfile('C:/tsLocal/libs/fundamentals/Systematic/latest.html',
                       save_path + 'latest.html')
       

   if str_title is None:
       str_title = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
   else:
       str_title = str_title + '// ' + \
           pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

   if file_prefix is None:
       # file_prefix = save_path + Strings.slugify(str_title)
       file_prefix = Strings.slugify(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
   else:
       file_prefix = file_prefix + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')   

    
   img_links = []         
   for fig_name, fig in figs.items():
       
       full_filename = f'{file_prefix}_{fig_name}.png'
               
       tsv.savePlot(fig, f'{save_path}{full_filename}')    
       img_links.append(full_filename)
           



   # img_links.append(Strings.slugify(str_title) + '_risk.png')   
   html_tail = """  <H3>%s</H3>  \n""" % str_title
   for imgn in img_links:
       html_tail += """ <img class="normal" src= "./%s" > \n""" % imgn
   html_tail += ' <p> '

   with open(save_path + 'latest.html', 'a') as w:
       w.write(html_tail) 
       

###############################################################################
def train_test_profit(df_quarter, target_sym_quarter, model, validation, train_first_len, 
                      val_len, train_len, test_len, selected_columns=None,
                      fixed_size_training=True, overlap_training=True, delta=0, 
                      overlap_trade=False, num_days=1, verbose=2, save_results=False, filename=None,
                      return_performance=False, return_index=False,  position_size_col=None):
    

   

    label_columns = get_label_columns(target_sym_quarter, num_days)    
       

    # print('label columns: ', label_columns)
    
    # print(df_quarter.columns)
    for col in label_columns:
        # print(col)
        if col not in df_quarter.columns:
            print(f'Error: Label column {col} is not in DataFrame')
            sys.exit()





    profit, y_test_cont_total, pred_test_total, train_test_index, mean_profit, num_positions, \
    sharpe, sortino = train_validation_test(df_quarter, 
                       model, validation, train_first_len, val_len, 
                       train_len, test_len, label_columns,
                       selected_columns=selected_columns,
                       fixed_size_training=fixed_size_training,
                       overlap_training=overlap_training, delta=delta, 
                       input_width=1, label_width=1, overlap_trade=overlap_trade, num_days=num_days, 
                       verbose=verbose, return_performance=True, return_index=True,  
                       position_size_col=position_size_col)

    
    if save_results:
        features = list(selected_columns) if selected_columns else list(df_quarter.columns)
        # save_model_results(filename, features, target_sym_quarter, model, validation, train_first_len, 
        #               val_len, train_len, test_len, fixed_size_training, overlap_training, delta, 
        #               overlap_trade, num_days, mean_profit, num_positions, sharpe, sortino)
        save_model_results(filename, features=[features],target_generator=target_sym_quarter,  
                            model=model.__class__.__name__,
                            model_parameters=[model.get_params()], validation=validation, 
                            train_first_len=train_first_len,  
                            val_len=val_len, train_len=train_len, test_len=test_len, 
                            fixed_size_training=fixed_size_training, overlap_training=overlap_training,
                            delta=delta, overlap_trade=overlap_trade, num_days=num_days,
                            mean_profit=mean_profit, num_positions=num_positions, sharpe=sharpe, 
                            sortino=sortino)
        
#                 'model parameters':[model.get_params()], 'validation':validation, 'train_first_len':train_first_len,
#                 'val_len':val_len, 'train_len':train_len, 'test_len':test_len, 
#                 'fixed_size_training':fixed_size_training, 'overlap_training':overlap_training,
#                 'delta':delta, 'overlap_trade':overlap_trade, 'num_days':num_days}
    
    if return_performance:
        if return_index:
            return profit, y_test_cont_total, pred_test_total, train_test_index, mean_profit, num_positions, sharpe, sortino    
        return profit, y_test_cont_total, pred_test_total, mean_profit, num_positions, sharpe, sortino   
    if return_index:
        return profit, y_test_cont_total, pred_test_total, train_test_index
    return profit, y_test_cont_total, pred_test_total







###############################################################################
# The function is based on validation rather than crosss-validation
def randomised_test_performance(df_input, target_sym_quarter, model, cols, noise, relative, num_real, train_len, test_len,  delta=0, num_days=1, position_size_col=None):
    l = train_len + test_len
    m = len(cols)
    df = df_input[:l].copy()
    
    if not isinstance(noise, (list, tuple, set)):
            noise = noise * np.ones(m)
    
    if relative:        
        df_change = df - df.shift(1)            
        noise_strength = noise * df_change[1:][cols].std().values    
    else:
        noise_strength = noise
    
    df_random = df.copy() 
    total_mean_profit, total_num_positions, total_sharpe, total_sortino = 0, 0, 0, 0
    
    for i in range(num_real):
        random_noise = np.random.uniform(-0.5, 0.5, size=(test_len, m)) * noise
        df_random[-test_len:][cols] = df[-test_len:][cols] + random_noise
        
        profit, y_test_cont_total, pred_test_total, mean_profit, num_positions, sharpe, sortino = train_test_profit(df_random, target_sym_quarter, model, 
                                                                        validation=True, train_first_len=train_len, 
                                                                        val_len=test_len, train_len=1, test_len=1, 
                                                                        fixed_size_training=True, overlap_training=True, 
                                                                        delta=delta, overlap_trade=False, num_days=num_days, 
                                                                        verbose=0, return_performance=True,
                                                                        position_size_col=position_size_col)
        
        # profit, mean_profit, num_positions, sharpe, sortino  =  day_profit(y_test_cont_total, pred_test_total,
        #                                                                 overlap_trade=False, num_days=num_days,
        #                                                                 verbose=0) 
        total_mean_profit += mean_profit
        total_num_positions += num_positions
        total_sharpe += sharpe
        total_sortino += sortino
      
    
        
    total_mean_profit /= num_real
    total_num_positions /= num_real
    total_sharpe /= num_real
    total_sortino /= num_real
    
    total_num_positions = int(total_num_positions)
    
    return total_mean_profit, total_num_positions, total_sharpe, total_sortino, noise_strength

###############################################################################
# The function is based on crosss-validation
def randomised_cross_validation_performance(X, y, y_cont, model, n_splits, cols, noise, relative, 
                                num_real, delta, num_days, target_sym=None, 
                                filename=None, verbose=0, position_size=None):
    
    
    # X_copy = X.copy()
    l = len(y)
    m = len(cols)
    
    if not isinstance(noise, (list, tuple, set)):
            noise = noise * np.ones(len(cols))
    
    if relative:        
        X_change = X - X.shift(1)            
        noise_strength = noise * X_change[1:][cols].std().values    
    else:
        noise_strength = noise
    
    X_random = X.copy() 
    total_mean_profit, total_num_positions, total_sharpe, total_sortino = 0, 0, 0, 0
    
   
    
    # print(X.shape, y.shape)
    # print(X.iloc[0], y.iloc[0])
    # print(f'Loop over {n_splits} train/test splits')
    
    # loop over random realisations
    for j in range(num_real):
        random_noise = np.random.uniform(-0.5, 0.5, size=(l, m)) * noise_strength
        X_random[cols] = X[cols] + random_noise
        print('j = ', j)
        
        y_test_total = np.array([])   
        pred_test_total = np.array([])
        binary_pred_test_total = np.array([])
        y_test_cont_total = np.array([])  
        test_ind = np.array([], dtype=int)   
        position_size_test_total = np.array([]) 
        
        cv_generator = split_generator(l=len(X), n_splits=n_splits, num_days=num_days)
        
        # loop over cross-validation splits
        for i, (train, test) in enumerate(cv_generator):  # loop over different train/test splits
            print('i =', i)
            X_train, y_train, = X.iloc[train,:], y.iloc[train] 
            X_test, y_test, y_test_cont = X_random.iloc[test, :], y.iloc[test],  y_cont.iloc[test].values
            # print(f'Train length: {len(train)}, test length: {len(test)}')
            position_size_test = position_size[test] if position_size is not None else 1
            # print('Position size test: ', position_size_test[:5])
            model.fit(X=X_train, y=y_train)
    
            binary_pred_test = model.predict(X_test)
            print_accuracy(binary_pred_test, y_test)
    
    
            th_min, th_max = 0.5 - delta, 0.5 + delta 
     
            pred_proba_test = model.predict_proba(X_test)
            pred_test = prob_to_label(pred_proba_test[:,1], th_min, th_max)
            profit, mean_profit, num_positions, sharpe, sortino  =  day_profit(y_test_cont, pred_test, 
                                                                               num_days=num_days, position_size=position_size_test, 
                                                                               verbose=0)
          
            ##############################################
            # Concatenate results with the previous ones
    
            y_test_total = np.concatenate((y_test_total, y_test))
            pred_test_total = np.concatenate((pred_test_total, pred_test))
            binary_pred_test_total = np.concatenate((binary_pred_test_total, binary_pred_test))
            y_test_cont_total = np.concatenate((y_test_cont_total, y_test_cont))
            test_ind = np.concatenate((test_ind, test))
            
            if not isinstance( position_size_test, (int, float)):
                  position_size_test_total = np.concatenate((position_size_test_total,  position_size_test))
            else:
                  position_size_test_total = 1

        ##############################################

        print('\nTOTAL\n') 
    
        print_accuracy(binary_pred_test_total, y_test_total)
        profit, mean_profit, num_positions, sharpe, sortino  =  day_profit(y_test_cont_total, pred_test_total,
                                                                           num_days=num_days, position_size=position_size_test_total,
                                                                           verbose=verbose)
   
    
        total_mean_profit += mean_profit
        total_num_positions += num_positions
        total_sharpe += sharpe
        total_sortino += sortino
      
    
        
    total_mean_profit /= num_real
    total_num_positions /= num_real
    total_sharpe /= num_real
    total_sortino /= num_real
    
    total_num_positions = int(total_num_positions)
    
    return total_mean_profit, total_num_positions, total_sharpe, total_sortino, noise_strength

###############################################################################
# return number of changes in array  
def num_changes(a):
    return len(np.where(a[:-1] != a[1:])[0])

def is_middle_opposite(a):
    if (a[0] == a[2]) and (a[0] != a[1]):
        return True
    return False

def not_all_equal(a):
    if (a[0] == a[1]) and (a[1] == a[2]):
        return False
    return True

def stability_test_performance(df_input, target_sym_quarter, model, cols, epsilon, \
                               relative, num_points, train_len, test_len,  delta=0, num_days=1,\
                               position_size_col=None):
    l = train_len + test_len
    m = len(cols)
    df = df_input[:l].copy()
    
    if not isinstance(epsilon, (list, tuple, set)):
            epsilon = epsilon * np.ones(m)
    
    predictions = {col: np.zeros((test_len - num_days, num_points)) for col in cols}
    
    for i, col in enumerate(list(cols)):
        
        print(col)
        if relative:        
            df_diff = df - df.shift(-1)   
            change = epsilon[i] * df_diff[[col]][-test_len:]
            # print
        else:
            change = epsilon[i] * np.ones((test_len, 1))
        
        for k in range(num_points):
        
            factor = k - (num_points - 1) / 2 
    
            df_changed = df.copy() 
            
            # print(len(df_changed[-test_len:][[col]]), len(factor * change))
            # df_changed[-test_len:][[col]] = df[-test_len:][[col]]
    
            df_changed[-test_len:][[col]] = df[-test_len:][[col]] + factor * change
            
            profit, y_test_cont_total, pred_test_total = \
                train_test_profit(df_changed, target_sym_quarter, model, 
                                validation=True, train_first_len=train_len, 
                                val_len=test_len, train_len=1, test_len=1, 
                                fixed_size_training=True, overlap_training=True, 
                                delta=delta, overlap_trade=False, num_days=num_days, 
                                verbose=0, position_size_col=position_size_col)
 
            # profit, mean_profit, num_positions, sharpe, sortino  =  day_profit(y_test_cont_total, pred_test_total,
            #                                                                 overlap_trade=False, num_days=num_days,
            #                                                                 verbose=0) 
            
            predictions[col][:, k] = pred_test_total
        
    stability_df = pd.DataFrame()
    
    for col in cols:
        for i, a in enumerate(predictions[col]):
            stability_df.loc[i, col] = num_changes(a)
            
    
    return stability_df



###############################################################################

def validation_performance(df, target_sym, model, total_val_len, train_len, test_len, 
                      selected_columns, delta, num_days, filename=None, verbose=0, position_size_col=None):
    
    save_results=False if filename is None else True
        
    df_val = df[:total_val_len] 
    
        
    profit, y_test_cont_total, pred_test_total, train_test_index,\
    mean_profit, num_positions, sharpe, sortino = train_test_profit(df_val, target_sym, model,
                                                                validation=False, train_first_len=1, 
                                                                val_len=1, train_len=train_len, test_len=test_len, 
                                                                selected_columns=selected_columns,
                                                                fixed_size_training=True, 
                                                                overlap_training=True, 
                                                                delta=delta, overlap_trade=False, 
                                                                num_days=num_days, verbose=verbose,
                                                                save_results=save_results, filename=filename,
                                                                return_performance=True,  return_index=True,
                                                                position_size_col=position_size_col)
    
    
    return profit, y_test_cont_total, pred_test_total, train_test_index, mean_profit, num_positions, sharpe, sortino




###############################################################################

def split_generator(l, n_splits, num_days):
    if l - num_days < n_splits:
        print(f'Error: the length {l} is not sufficient to make {n_splits} splits')
        return
    test_len = int(l / n_splits)
    for i in range(n_splits):
        test_ind = np.arange(i * test_len, (i + 1) * test_len)
        
        # if i == 0:
        #     train_ind = np.arange(test_len + num_days, l)
        #     yield train_ind, test_ind
        
        # if i == n_splits - 1:
        #     train_ind =  np.arange(0, l - test_len)
        #     yield train_ind, test_ind
        
        # if 0 < i < 
        
        train_ind1 = np.arange(0, i * test_len) 
        train_ind2 = np.arange((i + 1) * test_len + num_days, l)
        train_ind = np.concatenate((train_ind1, train_ind2))
        yield train_ind, test_ind
            
###############################################################################

def cross_validation_performance(X, y, y_cont, model, n_splits, delta, num_days, target_sym=None, filename=None, 
                                 verbose=0, position_size=None):
    
    cv_generator = split_generator(l=len(X), n_splits=n_splits, num_days=num_days)
    
    # print(X.shape, y.shape)
    # print(X.iloc[0], y.iloc[0])
    print(f'Loop over {n_splits} train/test splits')

    y_test_total = np.array([])   
    pred_test_total = np.array([])
    binary_pred_test_total = np.array([])
    y_test_cont_total = np.array([])  
    test_ind = np.array([], dtype=int)   
    position_size_test_total = np.array([]) 
    

  
    for i, (train, test) in enumerate(cv_generator):  # loop over different train/test splits
        print('i =', i)
        X_train, y_train, = X.iloc[train,:], y.iloc[train] 
        X_test, y_test, y_test_cont = X.iloc[test, :], y.iloc[test],  y_cont.iloc[test].values
        print(f'Train length: {len(train)}, test length: {len(test)}')
        position_size_test = position_size[test] if position_size is not None else 1
        # print('Position size test: ', position_size_test[:5])
        model.fit(X=X_train, y=y_train)

        binary_pred_test = model.predict(X_test)
        print_accuracy(binary_pred_test, y_test)


        th_min, th_max = 0.5 - delta, 0.5 + delta 

        pred_proba_test = model.predict_proba(X_test)
        pred_test = prob_to_label(pred_proba_test[:,1], th_min, th_max)
        model_performance  =  day_profit(y_test_cont, pred_test, num_days=num_days,
                                         position_size=position_size_test, verbose=verbose)
        
        profit, mean_profit, num_positions, sharpe, sortino = model_performance.values()
        ##############################################
        # Concatenate results with the previous ones

        y_test_total = np.concatenate((y_test_total, y_test))
        pred_test_total = np.concatenate((pred_test_total, pred_test))
        binary_pred_test_total = np.concatenate((binary_pred_test_total, binary_pred_test))
        y_test_cont_total = np.concatenate((y_test_cont_total, y_test_cont))
        test_ind = np.concatenate((test_ind, test))
        
        if not isinstance( position_size_test, (int, float)):
             position_size_test_total = np.concatenate((position_size_test_total,  position_size_test))
        else:
             position_size_test_total = 1

        ##############################################

    print('\nTOTAL\n') 

    print_accuracy(binary_pred_test_total, y_test_total)
    model_performance = day_profit(y_test_cont_total, pred_test_total, num_days=num_days, 
                                  position_size=position_size_test_total, verbose=verbose)
    
    
    profit, mean_profit, num_positions, sharpe, sortino = model_performance.values()
    
    
    if filename is not None:
        features = list(X.columns)
       
        save_model_results(filename, features=[features],target_generator=target_sym,  
                            model=model.__class__.__name__,
                            model_parameters=[model.get_params()], 
                            train_len=len(train), test_len=len(test), 
                            delta=delta, num_days=num_days,
                            mean_profit=mean_profit, num_positions=num_positions, sharpe=sharpe, 
                            sortino=sortino)
    
    return profit, y_test_cont_total, pred_test_total, test_ind, mean_profit, num_positions, sharpe, sortino


###############################################################################

def create_perturbed_df(X, cols_to_change, num_points, epsilon, change_mode, roll_window,\
                        group_mode=False):
    '''
    Returns DataFrame with multiindex containing perturbed X.  If group_mode = False, each feature from 
    cols_to_change is perturbed individually and the multiindex has the form [X.index, cols_to_change, shocks].
    If group_mode = True, all features from 
    cols_to_change are perturbed simultaneously and the multiindex has the form 
    [X.index, shocks].

    Parameters
    ----------
    X : DataFrame
        Input dat
    cols_to_change : list of str
        Columns to perturb 
    num_points : int
        number of perturbation points
        if num_points is odd the middle point is the unperturbed one  
    epsilon : int or list of int
        strength of perturbation, can be different for different columns
    change_mode : str
        'difference' : epsilon is multiplied by the absolute difference of
        the column change w.r.t. to the previous row
        'std' :  epsilon is multiplied by std of the column from a rollig window 
                of size roll_window 
        'absolute' :  epsilon is the same for all rows  
    roll_window : int
        size of a rollig window for change_mode='std'
    group_mode : boolean. Default is False
        if True, all features from cols_to_change are perturbed simultaneously
        if False, all features from cols_to_change are perturbed indivudually
        

    Returns
    -------
    DataFrame
    '''
    
    assert change_mode in ('difference', 'std', 'absolute')
    
    l = len(X)  # number of rows in the original DataFrame
    m = len(cols_to_change)  # number of columns to change
    
    if not isinstance(epsilon, (list, tuple, set)):
            epsilon = epsilon * np.ones(m)
        
      
    X_diff = np.abs(X[cols_to_change] - X[cols_to_change].shift(-1))   
    X_std = X[cols_to_change].rolling(roll_window).std()
    
    factors = np.arange(num_points) - (num_points - 1) / 2 
    
    if group_mode:
        index = pd.MultiIndex.from_product([X.index, factors], names=['date', 'shocks'])
    else:
        index = pd.MultiIndex.from_product([X.index, cols_to_change, factors], names=['date', 'features', 'shocks'])
    
    df_perturbed = pd.DataFrame(index=index, columns=X.columns)
    
    if group_mode:
        if change_mode == 'difference':                      
            change = epsilon * X_diff[cols_to_change].values
            # print
        elif change_mode == 'absolute':
            change = epsilon * np.ones((l, m))
            
        elif change_mode == 'std':
            change = epsilon * X_std[cols_to_change].values
        for factor in factors: 
                    print('factor =', factor)                  
                    X_changed = X.copy()     
                    X_changed[cols_to_change] = X[cols_to_change] + (factor * change)
                    df_perturbed.loc(axis=0)[:, factor] = X_changed.values 
        
    else:    
        for i, col in enumerate(cols_to_change):                
            print(col)
            if change_mode == 'difference':                      
                change = epsilon[i] * X_diff[[col]].values
                # print
            elif change_mode == 'absolute':
                change = epsilon[i] * np.ones((l, 1))
                
            elif change_mode == 'std':
                change = epsilon[i] * X_std[[col]].values
            
            for factor in factors:              
                X_changed = X.copy()     
                X_changed[[col]] = X[[col]] + factor * change
                df_perturbed.loc(axis=0)[:, col, factor] = X_changed.values

    return df_perturbed

###############################################################################


def perturbed_prob_predict(model, X, cols_to_change, num_points, epsilon, change_mode, roll_window,\
                           group_mode=False):
    '''
    Use create_perturbed_df() to create perturbed DataFrame,
    then make probability predictions for all rows in it,  added the result as a column and return it.


    Parameters
    ----------
    model : classifier
        model to make predictions     
    X : DataFrame
        Input data
    cols_to_change : list of str
        Columns to perturb 
    num_points : int
        number of perturbation points
        if num_points is odd the middle point is the unperturbed one  
    epsilon : int or list of int
        strength of perturbation, can be different for different columns
    change_mode : str
        'difference' : epsilon is multiplied by the absolute difference of
        the column change w.r.t. to the previous row
        'std' :  epsilon is multiplied by std of the column from a rollig window 
                of size roll_window 
        'absolute' :  epsilon is the same for all rows  
    roll_window : int
        size of a rollig window for change_mode='std'        
    group_mode : boolean. Default is False
       if True, all features from cols_to_change are perturbed simultaneously
       if False, all features from cols_to_change are perturbed indivudually    

    Returns
    -------
    DataFrame
    '''
    
    
    df_perturbed = create_perturbed_df(X, cols_to_change, num_points, epsilon, change_mode, roll_window)
    
    df_perturbed.dropna(inplace=True)
    
    pred = model.predict_proba(df_perturbed)
    num_classes = pred.shape[1]
    pred_col_names = [f'Prob_prediction_{i}' for i in range(num_classes)] 
    df_perturbed[pred_col_names] = pred
    
    return  df_perturbed

###############################################################################

def stability_cross_validation(X, y, model, n_splits, cols_to_change, num_points, \
                               epsilon, change_mode, roll_window, num_days, filename=None):
    '''
    
     Use perturbed_prob_predict() for cross-validation splits.
     Return probibility predictions for perturbed input. 
    

    Parameters
    ----------
    X : DataFrame
        X_test
    y : DataFrame
        y_test
    model : classifier
        model to make predictions    
    n_splits : int
        numner of cross-validation splits
     cols_to_change : list of str
        Columns to perturb 
    num_points : int
        number of perturbation points
        if num_points is odd the middle point is the unperturbed one  
    epsilon : int or list of int
        strength of perturbation, can be different for different columns
    change_mode : str
        'difference' : epsilon is multiplied by the absolute difference of
        the column change w.r.t. to the previous row
        'std' :  epsilon is multiplied by std of the column from a rollig window 
                of size roll_window 
        'absolute' :  epsilon is the same for all rows  
    roll_window : int
        size of a rolling window for change_mode='std'  
    num_days : int
       number of trading days
    filename : str, optional
        name file to save the output. The default is None.

    Returns
    -------
    prob_predictions : DataFrame
    Probibility predictions for perturbed input.   

    '''
   
    
    cv_generator = split_generator(l=len(X), n_splits=n_splits, num_days=num_days)
    
    X_perturbed = create_perturbed_df(X, cols_to_change, num_points, epsilon, change_mode, roll_window)
                          
    X_perturbed.dropna(inplace=True)
    
    X_perturbed_date_index = X_perturbed.index.unique(level=0)
    
    cv_generator = split_generator(l=len(X_perturbed_date_index), n_splits=n_splits, num_days=num_days)
    
    # y_extended = pd.Series(index=X_perturbed.index)
    
    # y_extended = y_extended.reset_index(level=0)['level_0'].map(y).values # extend y values to multiindex of X
    # # y_extended = y.to_frame().assign(ind1=ind1, ind2=ind2).set_index(['ind1', 'ind2'], append=True).iloc[:, 0]  
    
    
    prob_predictions =  pd.DataFrame(index=X_perturbed.index)

            
    # loop over cross-validation splits
    print(f'\nLoop over {n_splits} cross-validation splits')
    for i, (train, test) in enumerate(cv_generator):  # loop over different train/test splits
        print('i =', i)
        X_train, y_train, = X.iloc[train,:], y.iloc[train] 
       
        test_date_index = X_perturbed_date_index[test]        
        X_test = X_perturbed.loc(axis=0)[test_date_index,:, :]
                      
                                      
        model.fit(X=X_train, y=y_train)

        pred_proba_test = model.predict_proba(X_test)
    
        # print(X_test.shape) 
        # print(X_test.index)
    
      
        
        num_classes = pred_proba_test.shape[1]
        pred_col_names = [f'Prob_prediction_{i}' for i in range(num_classes)] 
        prob_predictions.loc[X_test.index, pred_col_names] = pred_proba_test
    
    
    prob_predictions.dropna(inplace=True)
    
    if filename is not None:
        prob_predictions.to_csv(filename)
    
    return prob_predictions


###############################################################################    

def group_index(index, method='winter_summer', **kwargs):
    """
    Split index into groups and return them

    Parameters
    ----------
    index : index-like
        index to be split into groups 
    method : str, optional
        method of splitting 
        'winter_summer' : index is pd.DatetimeIndex,
        the goups are formed by (year, season), where season = winter or summer
        'from_index_dic' : index_dic = {group_lavel: group_index} must be
        provided as an addtional argument; the groups are formed by
        group_label.
        The default is 'winter_summer'.
    
    **kwargs : optional keywords
    
    index_dic : index dictionary require for method='from_index_dic'
        

    Returns
    -------
    groups : TYPE
        DESCRIPTION.

    """
    if method == 'winter_summer':
        assert isinstance(index, pd.DatetimeIndex), 'index must be pd.DatetimeIndex'
        # summer_months = [4,5,6,7,8,9]
        winter_months_1 = [1,2,3]
        winter_months_2 = [10,11,12]
          
        df = pd.DataFrame(index=index)
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['season']  = 's'
        df.loc[df['month'].isin(winter_months_1), 'season'] = 'w1'
        df.loc[df['month'].isin(winter_months_2), 'season'] = 'w2'
        # group together e.g. last 3 months of 2020 and first 3 months of 2021
        df.loc[df['season'] == 'w1', 'year'] -= 1  
        df['season'] = np.where(df['season'] == 's', 's', 'w')
        groups = df.groupby(['year', 'season']).groups
        
    if method == 'from_index_dic':
        index_dic = kwargs.get('index_dic', None)
        assert index_dic is not None, 'index_dic is not provided'
    
        df = pd.DataFrame(index=index)
        df['index_group'] = np.nan
        for index_key, index_values in index_dic.items():
            df.loc[index_values,'index_group'] = index_key         
        groups = df.groupby(['index_group']).groups  
    return groups
        
    
 
  ##############################################################################    
# Convert regression prediction for returns into position size
def reg_prediction_to_pos_size(y, transform='abs_normalised', **kwargs):
    
    if transform == 'abs_normalised':
        pos_size = abs(y) 
        # normalise positions, so that <abs(pos_size)> = 1
        # pos_size /= pos_size.mean()
       
    return pos_size





  ##############################################################################    
# Decision statistics of meta model
def meta_label_decision_stats(tti, input_set='test', verbose=1):
      
    y_set = tti.set_dic[input_set]['y']
    df = tti.df
    
    y_set['primary_y_binary'] = df['primary_y_binary'][y_set.index]
    y_set['primary_binary_pred'] = df['primary_binary_pred'][y_set.index]
    y_set['primary_decision'] = df['primary_decision'][y_set.index]
    y_set['meta_label'] = df['meta_label'].shift(-tti.num_days)[y_set.index]
    
   # confusion_matrix(y_set['y_binary'], y_set['meta_label_pred'])
   
    # fraction of decisions which are set to zero (and were initially non zero) by meta-model  
    changed_frac = ((y_set['meta_label_pred'] == 0) & (y_set['primary_decision'] != 0)).sum() / len(y_set['meta_label_pred']) 
    # meta-labeks for such decisions
    meta_label_changed_decision = y_set['y_binary'][(y_set['meta_label_pred'] == 0) & (y_set['primary_decision'] != 0)]
    # fraction of such decisons which were intially wrong
    primary_wrong_changed_decison_frac = (meta_label_changed_decision == 10).sum() / len(meta_label_changed_decision)  
    meta_stat = {'changed_frac': changed_frac, 'primary_wrong_frac': primary_wrong_changed_decison_frac}
    
    if verbose > 0:
        print(f'Fraction of decisions which are set to zero (and were initially non zero) by meta-model {changed_frac:.2f}')
        print(f'Fraction of such decisons which were intially wrong  {primary_wrong_changed_decison_frac:.2f}')
    return meta_stat


 ##############################################################################    
# Exclude rows of df, for which the values of some featurte are from intervals
def exclude_intervals(df_input, col, intervals):
    df = df_input.copy()
    for (l_bound, r_bound) in intervals:
        df = df[(df[col] <= l_bound) | (df[col] > r_bound)]
    return df   
      
 ##############################################################################    
# Return index for which the values of some featurte are from intervals
def index_excluded_intervals(df_input, col, intervals):
    index_included = exclude_intervals(df_input, col, intervals).index
    index = df_input.index
    index_excluded = index[~index.isin(index_included)]
    return index_excluded  


 ##############################################################################    
# Return values corresponding to given percentiles for series
def percentile_values(s, percentiles):
    ranked = s.rank(pct=True).sort_values()
    perc_values = pd.Series(index=percentiles, dtype=s.dtype, name='value') 
    perc_values.index.rename('percentile', inplace=True)
    for p in percentiles: 
        idx = (ranked - float(p) / 100).abs().idxmin()
        perc_values[p] = s[idx]
        
    return perc_values                   
            
