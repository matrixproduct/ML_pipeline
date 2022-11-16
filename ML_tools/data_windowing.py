# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:18:16 2021


Module containing functions for

--- data windowing from time series

--- generating multiple train/test sets 

Details about WindowGenerator can be found at  https://www.tensorflow.org/tutorials/structured_data/time_series

@author: AOssipov
"""





import matplotlib as mpl
import numpy as np
import pandas as pd
import tensorflow as tf


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False



class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                  train_df, val_df, test_df,
                  label_columns=None, batch_size=32):
      # Store the raw data.
      self.train_df = train_df
      self.val_df = val_df
      self.test_df = test_df
      
      if train_df is not None and len(train_df) > 1: 
          self.train_first_ind = train_df.index[0]
          self.train_last_ind = train_df.index[-1 - shift] 
      if val_df is not None and len(val_df) > 1:
          self.val_first_ind = val_df.index[0]
          self.val_last_ind = val_df.index[-1 - shift] 
      else:
          self.val_first_ind = -1
          self.val_last_ind = -1
      if test_df is not None and len(test_df) > 1:    
          self.test_first_ind = test_df.index[0]
          self.test_last_ind = test_df.index[-1 - shift] 
      else:
          self.test_first_ind = -1
          self.test_last_ind = -1
    
      # Work out the label column indices.
      self.label_columns = label_columns
      if label_columns is not None:
        self.label_columns_indices = {name: i for i, name in
                                      enumerate(label_columns)}
      self.column_indices = {name: i for i, name in
                              enumerate(train_df.columns)}
    
      # Work out the window parameters.
      self.input_width = input_width
      self.label_width = label_width
      self.shift = shift
    
      self.total_window_size = input_width + shift
    
      self.input_slice = slice(0, input_width)
      self.input_indices = np.arange(self.total_window_size)[self.input_slice]
    
      self.label_start = self.total_window_size - self.label_width
      self.labels_slice = slice(self.label_start, None)
      self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
      self.batch_size = batch_size
    
    def __repr__(self):
      return '\n'.join([
          f'Total window size: {self.total_window_size}',
          f'Input indices: {self.input_indices}',
          f'Label indices: {self.label_indices}',
          f'Label column name(s): {self.label_columns}'])
    
    @tf.autograph.experimental.do_not_convert
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
          labels = tf.stack(
              [labels[:, :, self.column_indices[name]] for name in self.label_columns],
              axis=-1)
      
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
      
        return inputs, labels 
    
    # def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
    #     inputs, labels = self.example
    #     plt.figure(figsize=(12, 8))
    #     plot_col_index = self.column_indices[plot_col]
    #     max_n = min(max_subplots, len(inputs))
    #     for n in range(max_n):
    #       plt.subplot(max_n, 1, n+1)
    #       plt.ylabel(f'{plot_col} [normed]')
    #       plt.plot(self.input_indices, inputs[n, :, plot_col_index],
    #                label='Inputs', marker='.', zorder=-10)
      
    #       if self.label_columns:
    #         label_col_index = self.label_columns_indices.get(plot_col, None)
    #       else:
    #         label_col_index = plot_col_index
      
    #       if label_col_index is None:
    #         continue
      
    #       plt.scatter(self.label_indices, labels[n, :, label_col_index],
    #                   edgecolors='k', label='Labels', c='#2ca02c', s=64)
    #       if model is not None:
    #         predictions = model(inputs)
    #         plt.scatter(self.label_indices, predictions[n, :, label_col_index],
    #                     marker='X', edgecolors='k', label='Predictions',
    #                     c='#ff7f0e', s=64)
      
    #       if n == 0:
    #         plt.legend()
      
    #     plt.xlabel('Time [h]')
        
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size,)
      
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):  
      return self.make_dataset(self.train_df)
       
    @property
    def val(self):
      return self.make_dataset(self.val_df)
       
    @property
    def test(self):
      return self.make_dataset(self.test_df)
       
    @property
    def example(self):
      """Get and cache an example batch of `inputs, labels` for plotting."""
      result = getattr(self, '_example', None)
      if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
      return result
 

###########################################################################


def standard_scaling(train_df, test_df, val_df=None):
    train_mean = train_df.mean()
    train_std = train_df.std()
    
    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    if val_df is not None:
        val_df = (val_df - train_mean) / train_std
        
        return train_df, test_df, val_df
    
    return train_df, test_df


def list_from_dataset(dset):
    return list(dset.as_numpy_iterator())

def train_test_split(df, train_first_len, val_len, train_len, test_len, 
                     label_columns, fixed_size_training=True, overlap_training=True,
                     input_width=1, label_width=1, shift=1, batch_size=1, scaling=False, verbose=2):
    '''
    
   Generate multiple train/test sets represented by WindowGenerators compartible 
   with Tensorflow

    Parameters
    ----------
    df : DataFrame
         time series data
    train_first_len : int
        length of the first train set
    val_len : int
        length of the validation set
    train_len : int
        train set lenght
    test_len : int
     test set length
    label_columns : str or list of str
        column names for labels
    fixed_size_training : boolean, optional
        if True the leghts of train sets are fixed
        if False the train set is expanding 
        The default is True.
    overlap_training : boolean, optional
        if True a new train set overlaps with the previous one,
        such that a new test set starts where the previous test set ends
        if False the train sets don't overlap
        New train set starts where the previous train set ends
        Makes a difference only if fixe_size_training is True
        The default is True.    
    input_width : int, optional
        window input width, see the link above for details   
        The default is 1.
    label_width : int, optional
        window label width, see the link above for details   
        The default is 1.
    shift : int, optional
        window shift of labels, see the link above for details   
        The default is 1.
    batch_size : int, optional
        batch size for NN models.
        The default is 1.
    scaling : boolean, optional
        if True, the standard scaling is applied 
        The default is False.
    verbose : int
        verbosity mode
        The default is 2.
    Yields
    ------
    w_generators : generator of WindowGenerators
         

    '''
   
   
    
    if not isinstance(label_columns, (list, tuple, set)):
        label_columns = [label_columns]
    
    # w_generators = []  # list of windows generators
    train_start = 0
    train_end = train_start + train_first_len
    delta = min(1, shift)  # allowed overlap between train and val/test sets, such that label won't overlap 
    train_df_first = df[train_start: train_end]
    val_start = train_end - delta
    val_end = val_start + val_len
    val_df = df[val_start: val_end]
     
    # print('before 0 scaling')
    if scaling:
            train_df_first, val_df = standard_scaling(train_df_first, val_df)
    
    # print('after 0 scaling')
    w = WindowGenerator(input_width, label_width, shift,
                      train_df=train_df_first, val_df=None, test_df=val_df,
                      label_columns=label_columns, batch_size=batch_size)
    # print('after the first WindowGenerator')
    
    # w_generators.append(w)
    yield w
    
    # train_global_start = train_end
    train_global_start = max(val_end - train_len, train_end)
    train_start = max(val_end - train_len, train_end)  # start for new train set
    # test_start = val_start
    l = len(df)
    
   
    # print('before loop')
    # print(' new end:', train_start + train_len + test_len - delta, ' l:', l)
    # print(train_start, train_len, test_len)
    count = 0
    # while  train_start + train_len + test_len - delta < l:
    # cont_cond = (train_start + train_len + 2 * test_len - delta < l) if overlap_training else (train_start + 2 * train_len + test_len - delta < l)
    
    # adjust train and test lengths so that the actual len(y_train) = train_len and len(y_test) = test_len
    # train_len  += shift + input_width - 1
    # test_len  += shift + input_width - 1
    
    # if the lenght of remaining dataframe is smaller or equal than shift it should be included in the previous one 
    # cont_cond = (train_start + train_len + test_len - delta + shift + input_width < l) 
    cont_cond = (train_start + train_len + test_len - delta <= l) 
    
    #additional check for shift=0, as tf.keras.preprocessing.timeseries_dataset_from_array()
    # raises error in this case if len(df)=1
    if shift == 0 and (l - (train_start + train_len + test_len - delta)) == 1:
        cont_cond = False
    
    assert cont_cond, 'Train and test lenghts are too large compared to the total data available'
    # print('cont_cond:', cont_cond)
    
    
    
    # print(train_start, train_len, test_len, l)
    while cont_cond:    
        count += 1
        # print('\n Train start: ', train_start)
        train_end = train_start + train_len
        train_df = df[train_start: train_end]
        # print('len(train_df):', len(train_df))
        test_start = train_end - delta
        # print('Test start: ', test_start, '\n')
        test_end = test_start + test_len
        test_df = df[test_start: test_end]
        # print(f'Test len: {test_len}')
        if verbose > 0:
            print(f'\n\ntrain_start: {train_start}, train_end: {train_end}, test_start: {test_start}, test_end: {test_end}')

        
        ##############################################
        # Standard scaling
        # print('before scaling')
        if scaling:
            train_df, test_df = standard_scaling(train_df, test_df)
       
        # print(train_df)
        # print(df)
         
        # print('after scaling')
        # we are going to return w generated at the previous step in order 
        # to handle the last generator separately 
        # if count > 1: 
        #     w = deepcopy(w_next)
        # w_next = WindowGenerator(input_width, label_width, shift,
        #               train_df=train_df, val_df=None, test_df=test_df,
        #               label_columns=label_columns, batch_size=batch_size)
        
        w = WindowGenerator(input_width, label_width, shift,
                      train_df=train_df, val_df=None, test_df=test_df,
                      label_columns=label_columns, batch_size=batch_size)
        
        # w_generators.append(w)
        
        # if overlap_training()
        # new train and test sets:
        if fixed_size_training:
            # train_start = train_start + test_len - delta if overlap_training else test_start
            train_start = train_start + test_len - shift - input_width + 1  if overlap_training else \
                train_start + train_len - shift - input_width + 1
        else:
            train_start = train_global_start
            train_len = test_end - train_global_start
        # if count > 1:     
        #     yield w
        
        # cont_cond = (train_start + train_len + 2 * test_len - delta < l) if overlap_training else (train_start + 2 * train_len + test_len - delta < l)
        # cont_cond = (train_start + train_len + test_len - delta + shift + input_width < l) 
        cont_cond = (train_start + train_len + test_len - delta <= l) 
        
        #additional check for shift=0, as tf.keras.preprocessing.timeseries_dataset_from_array()
        # raises error in this case if len(df)=1
        if shift == 0 and (l - (train_start + train_len + test_len - delta)) == 1:
            cont_cond = False
            
        yield w    
    # handle the last possible train/test set separately, as we want it to end at the end of the dataframe
    
    # print(f'After loop Test len: {test_len}')
    # check if anything left in df and new train set can be created in case of non-overlap training
    if test_end < len(df) and (train_start + train_len - delta) < len(df): 
        train_end = train_start + train_len
        train_df = df[train_start: train_end]
        # print('len(train_df):', len(train_df))
        test_start = train_end - delta
        test_end = len(df)
        # print(f'After loop test_end: {test_end}')
        # print('Test start: ', test_start, '\n')
        test_df = df[test_start:]
        if verbose > 0:
            print(f'\n\ntrain_start: {train_start}, train_end: {train_end}, test_start: {test_start}, test_end: {test_end}')
    
        
        ##############################################
        # Standard scaling
        # print('before scaling')
        if scaling:
            train_df, test_df = standard_scaling(train_df, test_df)
       
        # print(train_df)
        # print(df)
         
        # print('after scaling')
       
        w = WindowGenerator(input_width, label_width, shift,
                      train_df=train_df, val_df=None, test_df=test_df,
                      label_columns=label_columns, batch_size=batch_size)
        
        yield w
          
    # return w_generators
    

        

###########################################################################

if __name__ == "__main__":
    
    # zip_path = tf.keras.utils.get_file(
    #     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    #     fname='jena_climate_2009_2016.csv.zip',
    #     extract=True)
    # csv_path, _ = os.path.splitext(zip_path)
    
    # df = pd.read_csv(csv_path)
    # # slice [start:stop:step], starting from index 5 take every 6th record.
    # df = df[5::6]
    
    # date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    
    
    # column_indices = {name: i for i, name in enumerate(df.columns)}
    
    # n = len(df)
    # train_df = df[0:int(n*0.7)]
    # val_df = df[int(n*0.7):int(n*0.9)]
    # val_df = None
    # test_df = df[int(n*0.9):]
    
    # num_features = df.shape[1]
    
    # w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
    #                       train_df=train_df, val_df=val_df, test_df=test_df,
    #                       label_columns=['T (degC)'])
    # print(w1)
    
    # w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
    #                      train_df=train_df, val_df=val_df, test_df=test_df,
    #                      label_columns=['T (degC)'])
    # print(w2)
    
    
    # # Stack three slices, the length of the total window:
    # example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
    #                            np.array(train_df[100:100+w2.total_window_size]),
    #                            np.array(train_df[200:200+w2.total_window_size])])
    
    # example_inputs, example_labels = w2.split_window(example_window)
    
    # # print('All shapes are: (batch, time, features)')
    # # print(f'Window shape: {example_window.shape}')
    # # print(f'Inputs shape: {example_inputs.shape}')
    # # print(f'labels shape: {example_labels.shape}')
    
    # # print(w2.train.element_spec)
    
    # # for example_inputs, example_labels in w2.train.take(1):
    # #   print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    # #   print(f'Labels shape (batch, time, features): {example_labels.shape}')
      
    # # print(w2.train) 
    
    
    # data = np.arange(100)
    # input_data = data[:-10]
    # targets = data[10:]
    # dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    #     input_data, targets, sequence_length=10)
    # for batch in dataset:
    # #   inputs, targets = batch
    # #   # assert np.array_equal(inputs[0], data[:10])  # First sequence: steps [0- 9]
    # #   # assert np.array_equal(targets[0], data[10])  # Corresponding target: step 10
      
    # #   print(inputs)
    # #   break  
    #     print(batch)
    
    # # print(list(dataset.as_numpy_iterator()))
    
    # split = train_test_split(df, train_first_len=5, val_len=3, train_len=4, test_len=2, 
    #                  label_columns=['T (degC)'], input_width=1, label_width=1, shift=1, scaling=True)
    
    df = pd.DataFrame({'f' : np.arange(15), 'label' : 10 * np.arange(15)})
    
    split = train_test_split(df, train_first_len=5, val_len=3, train_len=4, test_len=2, 
                     label_columns=['label'], fixed_size_training=True, input_width=1, label_width=1, shift=1, scaling=True)
    # split = list(split)
    
    # w0 = split[0]
    w0 = next(split)
    tr0, test0 = list_from_dataset(w0.train), list_from_dataset(w0.test)
    # w1 = split[1]
    # tr1, test1 = list_from_dataset(split[1].train), list_from_dataset(split[1].test)