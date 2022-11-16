# ML_tools

## Modules for efficient developing of new ML projects with the focus on time series data.
## The main class used in applications is TrainTest, see train_validation_test.py for details.

*******************************************************************************
data_windowing.py

Module containing functions for
--- data windowing from time series
--- generating multiple train/test sets 

Used in train_test_validation.py

*******************************************************************************
deep_train_test.py

Module containing functions for training, validating and testing for NN models.
Modification of class TrainTest for keras.

*******************************************************************************
features.py

Module containing functions for

--- feature selection/importance/clustering
--- bin selected features and get the statistics of 
other features w.r.t. that beans

*******************************************************************************
run_pytest.py

Script for running tests from test_train_validation_test.py

*******************************************************************************
test_train_validation_test.py

Test functions from train_validation_test.py

*******************************************************************************
train_validation_test.py


Module containing functions for training, validating and testing.
The main interface class is TrainTest; the objects of this class are used 
in scripts for training/testing models.

*******************************************************************************
