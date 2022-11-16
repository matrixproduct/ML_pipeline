# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:06:10 2022

@author: aossipov

Test functions from train_validation_test.py
"""

import numpy as np
import pandas as pd
import train_validation_test as tvt

def test_binary_return(): 
    x = np.array([-1, 0.5, 2, -0.1, 0])
    result = np.array([0, 1, 1, 0, 0])
    assert (tvt.binary_return(x) == result).all()


def test_prob_to_label(): 
    th_min, th_max = 0.3, 0.8
    x = np.array([0.1, 0.4, 0.9, 1, 0.3])
    result = np.array([-1, 0, 1, 1, 0])
    assert (tvt.prob_to_label(x, th_min, th_max) == result).all()
    
    th_min, th_max = 0.5, 0.5
    x = np.array([0.1, 0.4, 0.9, 1, 0.5])
    result = np.array([-1, -1, 1, 1, 1])
    assert (tvt.prob_to_label(x, th_min, th_max) == result).all
    
    
def test_max_drawdown():
    x = np.array([-1, 0.5, 5, 1])
    result = -4
    assert tvt.max_drawdown(x) == result
    
    x = np.array([-1, 0.5, 5, 6])
    result = -1
    assert tvt.max_drawdown(x) == result


def test_pnl_metric():
    profit = np.random.randn(100)
    num_days = 3
    num_positions = 15
    total_volume = 20.4
    pnl_metric_dic = tvt.pnl_metric(profit, num_days, num_positions, total_volume)
    assert isinstance(pnl_metric_dic, dict)
    
def test_day_profit():
    target_cont = np.random.randn(100)
    prediction = np.random.choice([-1, 0, 1], 100)
    model_performance = tvt.day_profit(target_cont, prediction, overlap_trade=False, 
                                    num_days=2, position_size=2.3, confluent_pos=True, 
                                    verbose=0)        
    assert isinstance(model_performance, dict)
    
    
def test_regression_metrics():
    target = np.random.randn(100)
    reg_prediction = np.random.randn(100)
    decision = np.random.choice([-1, 0, 1], 100)
    reg_metrics = tvt.regression_metrics(target, reg_prediction, decision)         
    assert isinstance(reg_metrics, dict)    


def test_classification_metrics():
    target = np.random.choice([0, 1], 100)
    prediction_prob = np.random.random(100)
    decision = np.random.choice([-1, 0, 1], 100)
    clf_metrics = tvt.classification_metrics(target, prediction_prob, decision)        
    assert isinstance(clf_metrics, dict)    


def test_group_index():
    index = pd.DatetimeIndex(['2020-10-01', '2022-03-01','2022-03-05','2020-11-05',\
                      '2020-07-12', '2021-01-01'])
    
    groups =  tvt.group_index(index, method='winter_summer')
    assert isinstance(groups, dict)  
    assert '2020-10-01' in groups[(2020, 'w')] and '2020-07-12' in groups[(2020, 's')]
    assert '2021-01-01' in groups[(2020, 'w')]     
    
    
def test_num_changes():    
    x = np.array([-1, -1, -1, 1, 1,-1, 1])
    result = 3
    assert tvt.num_changes(x) == result
    
    x = np.array([1, 1, 1, 1, 1, 1, -1])
    result = 1
    assert tvt.num_changes(x) == result
    
    
