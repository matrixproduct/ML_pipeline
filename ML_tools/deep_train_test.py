# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:23:14 2021

@author: aossipov

Module containing functions for training, validating and testing for NN models.

Modification of class TrainTest for keras.

"""

import numpy as np
# import pandas as pd
import tensorflow as tf

import random
try:
    import keras_tuner as kt
except ImportError:
    print('keras_tuner not installed')
# import pytest
# pytest.skip('This is not a test file', allow_module_level=True)


from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from joblib import dump

from data_windowing import train_test_split

from  train_validation_test import TrainTest, X_y_from_wgen, split_generator

##############################################################################
##############################################################################
# Generic Neural Network
class TrainTestNN(TrainTest):

    def __init__(self, df, model, validation, train_first_len, val_len, train_len, test_len,
                      label_columns, selected_columns=None, fixed_size_training=False,
                      overlap_training=True,  input_width=1, label_width=1, num_days=1, scaling=False,\
                      prediction_type='classification', trading=True, batch_size=1):

        super().__init__(df, model, validation, train_first_len, val_len, train_len, test_len,
                      label_columns, selected_columns, fixed_size_training,
                      overlap_training,  input_width, label_width, num_days, scaling,\
                      prediction_type, trading)


        self.models_weights = []  # we can't use deepcopy for NN models, therefore we
                                  # will save only their weights.

        self.batch_size = batch_size


    ##############################################################################
    # training model (and validation)
    def model_fit(self, x, y, validation_data=None, fit_parameters=None):
        if fit_parameters is None:
            fit_parameters = {}

        return self.model.fit(x, y, validation_data=validation_data, **fit_parameters)

   ###############################################
    # model prediction 
    def model_predict(self, x):
        y = self.model(x).numpy()
        y = y.reshape(-1)
        return y


    ##############################################################################
    # multiple models train and test for rolling window
    def train_test(self, create_x_y_test=True, tti_file=None, filename=None, fit_parameters=None,  verbose=0):



        # generator for train/test sets
        # print('before split')
        selected_and_label_columns = self.selected_columns.copy()
        for col in self.label_columns:
            if col not in selected_and_label_columns:
                selected_and_label_columns.append(col)

        split_gen = train_test_split(self.df[selected_and_label_columns], self.train_first_len, self.val_len, self.train_len,\
                                     self.test_len, self.label_columns, self.fixed_size_training,
                                     self.overlap_training, input_width=self.input_width,
                                     label_width=self.label_width, shift=self.num_days, verbose=verbose)


        # print('after split')

        if not self.validation:
              _ = next(split_gen)


        pred_test_total = np.array([]).reshape(0, self.num_classes) if self.prediction_type == 'classification' \
                          else np.array([])  # probability prediction for classification or prediciton for regression  for test set
    


        y_test_binary_total = np.array([])
        y_test_cont_total = np.array([])
        
       


        self.index.train = []
        self.index.val = []
        self.index.test = []
        history_total = []
        # train_test_index = {'train':[], 'val':[], 'test':[]}

        # train_dates = pd.Series([])
        # test_dates = pd.Series([])

        for w in split_gen:

            X_train, y_train_binary, y_train_cont, X_test, y_test_binary, y_test_cont = X_y_from_wgen(w)

        #     y_train, y_train_cont = y_train.values.flatten(), y_train_cont.values.flatten()
        #     y_test, y_test_cont = y_test.values.flatten(), y_test_cont.values.flatten()

            # print(X_train.shape, y_train.shape)



            ##############################################
            # Exclude some columns, if required
            # this is the only way not to include features used to generate
            # labels in the train/test sets

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
            # Model train and test
            if verbose > 1:
                print(f'\n Train dates: {(w.train_first_ind, w.train_last_ind)} \n')
                print(f'\n Test dates: {(w.test_first_ind, w.test_last_ind)} \n')




            if self.prediction_type == 'classification':
                history = self.model_fit(X_train, y_train_binary, validation_data=[X_test, y_test_binary], fit_parameters=fit_parameters)
                # history = self.model_fit(X_train, y_train_binary, fit_parameters=fit_parameters)

            else:
                history = self.model_fit(X_train, y_train_cont, validation_data=[X_test, y_test_cont], fit_parameters=fit_parameters)
                # history = self.model_fit(X_train, y_train_cont, fit_parameters=fit_parameters)

            history_total.append(history.history)
            self.models_weights.append(self.model.get_weights())



            ##############################################
            # Model prediction


            pred_test = self.model_predict_proba(X_test) if self.prediction_type == 'classification' else\
                        self.model_predict(X_test)
                                     



            ##############################################
            # Concatenate results with the previous ones


            # # if input_width > 1 we need to insert nans for y
            # # they correspond to the last (input_width - 1) rows for which we don't have y values

            # nan_arr = np.nan * np.ones(self.input_width - 1)
            # y_test_binary = np.append(y_test_binary, nan_arr)
            # y_test_cont = np.append(y_test_cont, nan_arr)
            # pred_test = np.append(pred_test, nan_arr)

            y_test_binary_total = np.concatenate((y_test_binary_total, y_test_binary))
            y_test_cont_total = np.concatenate((y_test_cont_total, y_test_cont))
            pred_test_total = np.concatenate((pred_test_total, pred_test))


            self.index.train.append((w.train_first_ind, w.train_last_ind))
            self.index.val.append((w.val_first_ind, w.val_last_ind))
            self.index.test.append((w.test_first_ind, w.test_last_ind))



            # self.pred_proba_test_total


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

        if tti_file is not None:
          dump(self, tti_file)


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


          self.X_y_test['X'].to_csv(test_X_file)
          self.X_y_test['y'].to_csv(test_y_file)

        return history_total


#############################################################################

    def cross_validation(self, n_splits, filename=None, verbose=0, fit_parameters=None, **kwargs):


        X_train = self.df[:self.train_len]
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
        history_total = []
        # position_size_test_total = np.array([])

        self.index.cross_val = []

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



            if self.prediction_type == 'classification':
                history = self.model_fit(X_train, y_train, validation_data=[X_test, y_test_binary], fit_parameters=fit_parameters)
            else:
                history = self.model_fit(X_train, y_train, validation_data=[X_test, y_test_cont], fit_parameters=fit_parameters)

            self.models_weights.append(self.model.get_weights())



            ##############################################
            # Model prediction

            pred_test = self.model_predict_proba(X_test.values) if self.prediction_type == 'classification' else\
                        self.model_spredict(X_test.values)




            ##############################################
            # Concatenate results with the previous ones

            y_test_binary_total = np.concatenate((y_test_binary_total, y_test_binary))
            y_test_cont_total = np.concatenate((y_test_cont_total, y_test_cont))
            pred_test_total = np.concatenate((pred_test_total, pred_test))
            history_total.append(history.history)


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


        # if tti_file is not None:
        #   dump(self, tti_file)


        if filename is not None:
          dump(self.X_y_cross_val, filename)


        return history_total


##############################################################################
##############################################################################
# Supervised Autoencoder combined with Multilayer Perceptron
class TrainTestSAE_MLP(TrainTestNN):

    ##############################################################################
    # training model (and validation)
    def model_fit(self, x, y, validation_data=None, fit_parameters=None):
        if fit_parameters is None:
            fit_parameters = {}
        validation_data_= None
        if validation_data is not None:
            x_test, y_test = validation_data
            validation_data_ = (x_test, y_test, y_test)

        return self.model.fit(x, [x, y, y],  validation_data=validation_data_, **fit_parameters, shuffle=True)

    
    ##############################################################################
    # model prediction
    def model_predict(self, x):

        x1, y1, y2 = self.model(x)
        y2 = y2.numpy().reshape(-1)
        # y1 = y1.numpy().reshape(-1)

        return y2  # MLP return
        # return y1  # AE return  
        
    ##############################################################################
    # model prediction probability
    def model_predict_proba(self, x):
        x1, y1, y2 = self.model(x)
       # y2 = y2.numpy().reshape(-1)
        y2 = y2.numpy().reshape(-1)
        # y1 = y1.numpy().reshape(-1)
        proba = np.stack((y2, 1 - y2), axis=-1)    
        return proba
        # return y   

     ###############################    
##############################################################################
##############################################################################
# Hyperparameter turning for cross-validation
class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, tti, n_splits, fit_parameters=None, **kwargs):

        tti.model = self.hypermodel.build(trial.hyperparameters)
        history = tti.cross_validation(n_splits, fit_parameters=fit_parameters, **kwargs)
        val_losses = np.array([[hist[k][-1] for k in hist] for hist in history])
        self.oracle.update_trial(trial.trial_id, {k:np.mean(val_losses[:,i]) for i,k in enumerate(history[0].keys())})
        # self.save_model(trial.trial_id, tti.model)







##############################################################################
##############################################################################
# It's imortant to  set all three seeds below ino reder to get reproducible results

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

##############################################################################
##############################################################################
