# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:36:18 2021

Module containing functions for

--- feature selection/importance/clustering
--- bin selected features and get the statistics of 
other features w.r.t. that beans

@author: aossipov
"""


##############################################################################
# Feature importance
##############################################################################
# Sources:
# 1.    "Machine Learning for Asset Managers (Elements in Quantitative Finance)"
#         by Marcos M López de Prado (Author)
    
    
# 2.    https://github.com/emoen/Machine-Learning-for-Asset-Managers

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
# from sklearn.utils import check_random_state
from sklearn.metrics import log_loss
from sklearn.model_selection._split import KFold
# from scipy.linalg import block_diag
import matplotlib.pylab as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.base import is_classifier

# from featurewiz import FE_remove_variables_using_SULOV_method


import sys

sys.path.append('C:/tsLocal/folib/ML_tools')

  
from train_validation_test import cross_validation_performance, get_label_columns, validation_performance



# import matplotlib

# import ch2_marcenko_pastur_pdf as mp

#Code snippet 6.1 generating a set of informative, redundant, and noisy explanatory variables
# returns matrix X of training samples, and vector y of class labels for the training samples
def getTestData(n_features=100, n_informative=25, n_redundant=25, n_samples=10000, random_state=0, sigmaStd=.0):
    '''
    Generate a random dataset for classification problem consiting of informative, redundant and noise features.
    Return samples and labels.

    Parameters
    ----------
    n_features : int, optional
        Total number of featuters. The default is 100.
    n_informative : int, optional
         Number of informative features. The default is 25.
    n_redundant : int, optional
        Number of redundant features. The default is 25.
    n_samples : int, optional
        Number of samples. The default is 10000.
    random_state : int, optional
        Random seed. The default is 0.
    sigmaStd : float, optional
        Variance of noise distribution for noise features. The default is .0.

    Returns
    -------
    X : DataFrame
        Samples
    y : Series
        Labels

    '''
    
    for n in (n_features, n_informative, n_redundant, n_samples, random_state):
        assert isinstance(n, int)
    
    np.random.seed(random_state)
    X, y = make_classification(n_samples=n_samples, n_features=n_features-n_redundant, 
        n_informative=n_informative, n_redundant=0, shuffle=False, random_state=random_state)
    cols = ['I_' + str(i) for i in range(0, n_informative)]
    cols += ['N_' + str(i) for i in range(0, n_features - n_informative - n_redundant)]
    X, y = pd.DataFrame(X, columns=cols), pd.Series(y)
    i = np.random.choice(range(0, n_informative), size=n_redundant)
    for k, j in enumerate(i):
        X['R_' + str(k)] = X['I_' + str(j)] + np.random.normal(size=X.shape[0]) * sigmaStd    
    return X, y 



'''
Optimal Number of Clusters (ONC Algorithm)
Detection of False Investment Strategies using Unsupervised Learning Methods
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017
'''

'''codesnippet 4.1
 base clustering: Evaluate the correlation matrix as distance matrix,
 the find cluster; in the inner loop, we try different k=2..N
 on which to cluster with kmeans for one given initialization,
 and evaluate q = E(silhouette)/std(silhouette) for all clusters.
 The outer loop repeats inner loop with initializations of
 _different centroid seeds_
  
 kmeans.labels_ is the assignment of members to the cluster
 [0 1 1 0 0]
 [1 0 0 1 1] is equivelant
'''
def clusterKMeansBase(corr0, maxNumClusters=10, n_init=10, debug=False, random_state=None):
    '''
    Cluster features using a distance function constructed from the correlation matrux and the k-means algorithm.
    The Silhouette score is used to determine the optimal number of clusters.
    Parameters
    ----------
    corr0 : array
        correlation matrix
    maxNumClusters : int, optional
        Maximal number of clusters. The default is 10.
    n_init : int, optional
        Number of intializations for the k-means algorithm. The default is 10.
    debug : boolean, optional
        If True, print out additional info. The default is False.
    random_state : int, optional
        Random seed. The default is None.

    Returns
    -------
    corr1 : DataFrame
        Correlation matrix with reordered rows and columns corresponding 
        to clusters
    clstrs : dictionary
        Clusters, clstrs[i] is a list of features in ith cluster
    silh_coef_optimal : Series
        Silhouette score for each feature

    '''
    
    for n in (maxNumClusters, n_init):
        assert isinstance(n, int)
    
    assert isinstance(corr0, pd.DataFrame)
     
    corr0[corr0 > 1] = 1
    dist_matrix = ((1 - corr0.fillna(0)) / 2.) ** .5  #observations matrix
    silh_coef_optimal = pd.Series(dtype='float64') 
    kmeans, stat = None, None
    maxNumClusters = min(maxNumClusters, int(np.floor(dist_matrix.shape[0]/2)))
    print("maxNumClusters:", maxNumClusters)
    if random_state is not None:
        np.random.seed(random_state)
    for init in range(0, n_init):
    #The [outer] loop repeats the first loop multiple times, thereby obtaining different initializations. Ref: de Prado and Lewis (2018)
    #DETECTION OF FALSE INVESTMENT STRATEGIES USING UNSUPERVISED LEARNING METHODS
        # for num_clusters in range(4, maxNumClusters+1):
        for num_clusters in range(2, maxNumClusters+1):    
            seed = np.random.choice(range(0, 1000 * n_init * maxNumClusters))
            #(maxNumClusters + 2 - num_clusters) # go in reverse order to view more sub-optimal solutions
            kmeans_ = KMeans(n_clusters=num_clusters, n_init=1, random_state=seed) #, random_state=3425) #n_jobs=None #n_jobs=None - use all CPUs
            kmeans_ = kmeans_.fit(dist_matrix)
            silh_coef = silhouette_samples(dist_matrix, kmeans_.labels_)
            stat = (silh_coef.mean() / silh_coef.std(), silh_coef_optimal.mean() / silh_coef_optimal.std())

            # If this metric better than the previous set as the optimal number of clusters
            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh_coef_optimal = silh_coef
                kmeans = kmeans_
                if debug==True:
                    print(kmeans)
                    print(stat)
                    silhouette_avg = silhouette_score(dist_matrix, kmeans_.labels_)
                    print(f"For n_clusters = {num_clusters} the average silhouette_score is {silhouette_avg}")
                    print("********")
    
    newIdx = np.argsort(kmeans.labels_)
    # print(newIdx)

    corr1 = corr0.iloc[newIdx] #reorder rows
    corr1 = corr1.iloc[:, newIdx] #reorder columns

    clstrs = {i:corr0.columns[np.where(kmeans.labels_==i)[0]].tolist() for i in np.unique(kmeans.labels_)} #cluster members
    silh_coef_optimal = pd.Series(silh_coef_optimal, index=dist_matrix.index)
    
    return corr1, clstrs, silh_coef_optimal


##############################################################################

#code snippet 6.5 - clustered MDA    
def featImpMDA_Clustered(clf, X, y, clstrs, n_splits=10, num_real=1, random_state=None):
    '''
    Use Mean Decrease Accuracy to calculate the cluster importance. 
    Return importance score for each cluster. 
    
    
    Parameters
    ----------
    clf : Classifier
        Model used to evaluate the feature importance
    X : DataFrame
        Samples
    y : DataFrame or Series
        Labels
    clstrs : dictionary
        Clusters, the same format as in output of clusterKMeansBase()
    n_splits : int, optional
        Number of splits for cross-validation. The default is 10.
    num_real : int, optional
        Number of random realisations. The default is 1.
    random_state : int, optional
        Random seed. The default is None.

    Returns
    -------
    imp : DataFrame
        Importance score

    '''
    assert is_classifier(clf)
    
    for n in (n_splits, num_real):
        assert isinstance(n, int)
    
    if random_state is not None:
        np.random.seed(random_state)
        print('random state:', random_state)
    cvGen = KFold(n_splits=n_splits)
    # scr0 (scr0) is a score without (with) shuffling clusters
    scr0, scr1 = pd.Series(dtype='float64'), pd.DataFrame(columns=clstrs.keys())
    print(f'Loop over {n_splits} train/test splits')
    for i, (train, test) in enumerate(cvGen.split(X=X)):  # loop over different train/test splits
        print('i =', i)
        X0, y0, = X.iloc[train,:], y.iloc[train] 
        X1, y1 = X.iloc[test, :], y.iloc[test]
        fit = clf.fit(X=X0, y=y0)
        prob=fit.predict_proba(X1)
        scr0.loc[i] = -log_loss(y1, prob, labels=clf.classes_)
        for j in scr1.columns:  # loop over clusters
            mean_res = 0 
            for _ in range(num_real):  # loop over random realisations for shuffling 
                X1_=X1.copy(deep=True)
                for k in clstrs[j]:  # loop over features in a cluster
                    np.random.shuffle(X1_[k].values) # shuffle clusters
                prob=fit.predict_proba(X1_)
                mean_res += -log_loss(y1, prob, labels=clf.classes_)
            scr1.loc[i,j] =  mean_res / num_real
        imp=(-1 * scr1).add(scr0,axis=0)
        imp = imp / (-1 * scr1)  # imp = 1 - scr0/scr1 
        imp = pd.concat({'mean':imp.mean(), 'std':imp.std()*imp.shape[0] ** -.5}, axis=1) # it's not clear why do we 
        # need to devide std() by sqrt(n_splits)?
        imp.index=['C_' + str(i) for i in imp.index]
    return imp




##############################################################################

def clustered_MDA_feature_imp(X, y, clf, maxNumClusters=10, n_init=10, n_splits=10, num_real=1, random_state=None):
    '''
    Group features in clusters using clusterKMeansBase(), then run featImpMDA_Clustered() to
    get the cluster importance. Return clusters and mean and std of their importance.

    Parameters
    ----------
    X : DataFrame
        train input samples
    y : DataFrame or array
        train labels
    clf : classifier
        model used for feature imporatnce
    maxNumClusters : int, optional
        maximum number of clusters 
        The default is 10.
    n_init : int, optional
        number of realisations for the clustering algorithm. 
        The default is 10.
    n_splits : int, optional
        Number of splits for the MDA algorithm 
        The default is 10.
    num_real : int, optional
        number of random realisations for shuffling 
        The default is 1.
    random_state : int, optional
        random seed for shuffling 
        The default is None.     
        

    Returns
    -------
    Dictionary, DataFrame

    '''
    assert is_classifier(clf)
    
    for n in (maxNumClusters, n_init, n_splits, num_real):
        assert isinstance(n, int)
    
    print('Finding clusters\n')
    corr0, clstrs, silh = clusterKMeansBase(X.corr(), maxNumClusters=maxNumClusters, n_init=n_init, random_state=random_state)
    # fit = clf.fit(X,y)
    print('Clusters:')
    for ind, values in clstrs.items():
        print(f'{ind}: {values}')
    print('\nApplying MDA')
    imp = featImpMDA_Clustered(clf, X, y, clstrs, n_splits=n_splits, num_real=num_real, random_state=random_state)
    imp.sort_values('mean', inplace=True)
   
    plt.figure(figsize=(10, 5))
    imp['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
    plt.title('Feature importance: Clustered MDA')
    plt.show()
    
    return clstrs, imp

#

#############################################################################

def feature_clustering_selection(df, target_sym, model, total_val_len, 
                                    val_train_len, val_test_len, 
                                    selected_columns, delta, num_days, 
                                    maxNumClusters=10, n_init=10, n_splits=10, num_real=200, random_state=None,
                                    feature_selection=True, select_metric='sortino',
                                    verbose=0):
    '''
    
    Group features in clusters and find their importance using clustered_MDA_feature_imp(), 
    then optioanlly remove clusters looking for the best performance. 
    Return clusters and mean and std of their importance.
    If  feature_selection is True additionaly return best clusters and features to remove.
     

    Parameters
    ----------
    df : DataFrame
        Inout data
    target_sym : str
        target generating key in df
    model : Classifier
        model used for clustering and feature selection
    total_val_len : int
        total length of the validation set
        df[: total_val_len] is used for train and test
    val_train_len : int
        train length, df[:train_len] is a train set
    val_test_len : int     
        test length, if val_train_len + val_test_len < total_val_len
        several test sets can be generated by validation_performance()
    selected_columns : list of str
        features to selecct from df         
    delta : float
        parameter used to convert probility model prediction to label
    num_days : int
        number of tarding days
    maxNumClusters : int, optional
        maximum number of clusters 
        The default is 10.
    n_init : int, optional
        number of realisations for the clustering algorithm. 
        The default is 10.
    n_splits : int, optional
        Number of splits for the MDA algorithm 
        The default is 10.
    num_real : int, optional
        number of random realisations for shuffling 
        The default is 200
    random_state : int, optional
        random seed for shuffling 
        The default is None.     
    feature_selection : boolean, optional
        If True feature selction is enabled. The default is True.
    select_metric : str, optional
        Metric to determine the best performance. The default is 'sortino'.
    verbose : int
        verbosity mode
        The default is 0.

    Returns
    -------
    clusters, impurity, best_clusters, exclude_columns, all_removed_columns
    
    exclude_columns: all features were excluded to achieve the best performance
    all_removed_columns: all  features were removed by this procedure
   
    clusters, impurity are 
    the same output as in clustered_MDA_feature_imp()
    
    if  feature_selection=False, best_clusters = clusters, exclude_columns, all_removed_columns =  [], []

    '''
        
    ##########################################################################
    # Clustering
    ##########################################################################
    print('\n*******  Feature clustering *******\n')
    
    label_columns =  get_label_columns(target_sym, num_days)
    # selected_columns = [f for f in df.columns if f not in exclude_columns]
    exclude_columns = [f for f in df.columns if f not in selected_columns]
    
    X_train = df[:total_val_len]
    y = X_train[label_columns[1]].shift(-num_days)[:-num_days]
    X = X_train[:-num_days][selected_columns] 
    
    maxNumClusters = min(maxNumClusters, len(selected_columns) - 1) 
    
    # print('\nSelected columns:', selected_columns)
    # print(X.shape, y.shape)
    clusters, impurity = clustered_MDA_feature_imp(X, y, model, maxNumClusters=maxNumClusters, n_init=n_init, n_splits=n_splits, num_real=num_real, random_state=random_state)
    _ = [print(f'Cluster: {key} \n{value}\n') for key, value in clusters.items()]
    print('impurity:\n', impurity)
    
    if not feature_selection:
        
        return clusters, impurity, clusters, [], []
    
    ##########################################################################
    # Exclude clusters
    ##########################################################################
    print('\n*******  Feature selection *******\n')
    
    if exclude_columns is None:
        exclude_columns = []
    all_removed_cols = []    
    
    profit, y_test_cont_total, pred_test_total, mean_profit, num_positions, sharpe, sortino =  validation_performance(df, target_sym, model, total_val_len, 
                                                                                                                    val_train_len, val_test_len, 
                                                                                                                    selected_columns, delta, num_days, 
                                                                                                                    filename=None, verbose=0)
                                                            
   
        
    best_performance = {'mean_profit': mean_profit, 'sharpe': sharpe, 'sortino': sortino}[select_metric]
    if verbose > -1:
        print(f'Best performance before loop: {select_metric}  {best_performance}')
        # bp = {'mean_profit': mean_profit, 'sharpe': sharpe, 'sortino': sortino}
        # print(f'Best performance before loop: {bp}')
        
    
   
    
    i = 0
    clusters_copy = clusters.copy()
    while len(clusters) > 1:
        if verbose > 0:
            print(f'Loop over clusters, run {i + 1}')
       
        cluster_ind = int(impurity.index[i].replace('C_', ''))
        exclude_columns_copy = exclude_columns.copy()
        exclude_columns += clusters[cluster_ind]
        new_selected_columns = [f for f in df.columns if f not in exclude_columns]
        
        profit, y_test_cont_total, pred_test_total, mean_profit, num_positions, sharpe, sortino =  validation_performance(df, target_sym, model, total_val_len, 
                                                                                                                    val_train_len, val_test_len, 
                                                                                                                    new_selected_columns, delta, num_days, 
                                                                                                                    filename=None, verbose=0)
        # print('Exclude columns:', exclude_columns)
        # print( {'mean_profit': mean_profit, 'sharpe': sharpe, 'sortino': sortino})
       
        performance = {'mean_profit': mean_profit, 'sharpe': sharpe, 'sortino': sortino}[select_metric]
        
        if performance >= best_performance:
            best_performance = performance
            if verbose > -1:
                print(f'Current best performance: {select_metric} {best_performance}')
            removed_cols = clusters.pop(cluster_ind)
            all_removed_cols += removed_cols
            if verbose > -1:
                print(f'Removed features {removed_cols}')
                # print(f'Remaining features to check  {cols}')
            i += 1
        else:
            exclude_columns = exclude_columns_copy
            break   
            
    
   
    if not all_removed_cols:
        print('No features were removed')
    else:
        print(f'\nFinal best performance: {select_metric} {best_performance}')
        print('All removed features: ',  exclude_columns)
        print('Features removed by this procedure : ',  all_removed_cols)
        # print('cols: ', cols)
    
    best_clusters = clusters
    clusters = clusters_copy

    return clusters, impurity, best_clusters, exclude_columns, all_removed_cols
##############################################################################


def plot_bin_one_target(df_stat):
    '''
    
    Plot data from df_stat

    Parameters
    ----------
    df_stat : DataFrame
        Statistics w.r.t. to a particular feature.
        Any element of a list returned by get_bin_stat
    selected_cols : str or list of str
       features to plot
    ncols : int, optional
        Number of columns for plots. The default is 4.

    Returns
    -------
    None.

    '''
    
        
    col_to_bin = df_stat.index.name
    stats = df_stat.columns.unique()
    ncols = len(stats)
    nrows= 1
    fig, ax = plt.subplots(nrows= nrows , ncols=ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(col_to_bin)

    for j, col in enumerate(stats):             
        x = [interv.mid for interv in df_stat.index.values]
        y = df_stat[col]
        ax[0,j].plot(x, y, label=col)
        ax[0, j].legend()
                
                    
##############################################################################
# Bin features, get statistics for just a target
##############################################################################



def get_bin_stat_target(df, cols_to_bin, target, bins, quantiles=True, best_to_plot = 0):
    '''
    
    Bin every feature from cols_to_bin in df  into bins
    then calculate mean, std, and skew for target
    feature in df w.r.t. that bins.
    Return a list of DataFrames with the corresponding statistics.

    Parameters
    ----------
    df : DataFrame
        Features
    cols_to_bin : str or list of str 
       Features to bin
    bins : int or array
        (Number of) bins. See pd.cut for more info.
    quantiles: boolean
        If True use pd.qcut for quantile-based discretization,
        otherwise use pd.cut to bin values. 
        The default is True.

    Returns
    -------
    df_bins_stat : list
        List of DataFrames containing statistics w.r.t. each feature
        from cols_to_bin

    '''

    if not isinstance(cols_to_bin, (list, tuple, set, pd.core.indexes.base.Index)):
        cols_to_bin = [cols_to_bin]

    bin_cols = []
    bin_cols_binary=[]
    
    df_copy = df.copy()
    diff_thresshold = 1/bins
    
    for col in cols_to_bin:
        print(col)
        num_values =  len(df[col].unique())
        if num_values == 1:
            print(f'WARNING: Column {col} contains a single value only and will be ignored')
            continue
        if isinstance(bins, int) and num_values < bins:
            print(f'WARNING: The number of bins will be reduced to {num_values}, which is the number of unique values for column {col}.')
            bin_cols_binary.append(col)

        else:     
            # df_copy[col + ' bins'] = pd.qcut(df[col], bins, duplicates='drop') if quantiles else pd.cut(df[col], bins=bins)
            # bin_cols.append(col + ' bins')
            df_copy[col] = pd.qcut(df[col], bins, duplicates='drop') if quantiles else pd.cut(df[col], bins=bins)
            bin_cols.append(col)


    df_stat_dict = {}
    target_stat = pd.DataFrame()
    target_stat_binary = pd.DataFrame()
    for col in bin_cols_binary:
        new_el = df_copy[[target,col]].groupby(col).agg(['mean', 'std', 'skew'])[target]      
        df_stat_dict[col] = new_el
        if (max(new_el['std'])==0):
            target_stat_binary.loc[col,'max_min'] = (max(new_el['mean'])-min(new_el['mean']))
        else:
            target_stat_binary.loc[col,'max_min'] = (max(new_el['mean'])-min(new_el['mean']))/max(new_el['std'])

        
    for col in bin_cols:
        new_el = df_copy[[target,col]].groupby(col).agg(['mean', 'std', 'skew'])[target]      
        df_stat_dict[col] = new_el
        target_stat.loc[col,'extreme_diff'] = abs(new_el.iloc[-1,0]-new_el.iloc[0,0])/max(new_el.iloc[-1,1],new_el.iloc[0,1])
        if (max(new_el['std'])==0):
            target_stat.loc[col,'max_min'] = (max(new_el['mean'])-min(new_el['mean']))
        else:
            target_stat.loc[col,'max_min'] = (max(new_el['mean'])-min(new_el['mean']))/max(new_el['std'])
        
        diff = pd.DataFrame()
        for i in range(0,len(new_el.index)-1):
            diff.loc[i,'first_diff'] = (new_el.iloc[i+1,0]- new_el.iloc[i,0])/max(new_el.iloc[i+1,1],new_el.iloc[i,1])
        diff['thresshold'] = 0
        diff.loc[diff['first_diff'] > diff_thresshold, 'thresshold'] = 1
        diff.loc[diff['first_diff'] < -diff_thresshold,'thresshold'] = -1
        diff['thresshold_ffill']= diff['thresshold'].replace(to_replace=0, method='ffill')
        diff['thresshold_dif'] = abs(diff['thresshold_ffill']-diff['thresshold_ffill'].shift(1))
        target_stat.loc[col,'switching']  =  len(diff[diff['thresshold_dif']==2])
    target_stat.sort_values(by=['max_min'], ascending=False, inplace=True)       
    selected_cols = target_stat.index[:best_to_plot]
    for feat in selected_cols:
        plot_bin_one_target(df_stat_dict[feat])

    return df_stat_dict, target_stat, target_stat_binary
##############################################################################
# Bin features, get statistics
##############################################################################



def get_bin_stat(df, cols_to_bin, bins, quantiles=True):
    '''
    
    Bin every feature from cols_to_bin in df  into bins
    then calculate mean, std, and skew for all
    features in df w.r.t. that bins.
    Return a list of DataFrames with the corresponding statistics.

    Parameters
    ----------
    df : DataFrame
        Features
    cols_to_bin : str or list of str 
       Features to bin
    bins : int or array
        (Number of) bins. See pd.cut for more info.
    quantiles: boolean
        If True use pd.qcut for quantile-based discretization,
        otherwise use pd.cut to bin values. 
        The default is True.

    Returns
    -------
    df_bins_stat : list
        List of DataFrames containing statistics w.r.t. each feature
        from cols_to_bin

    '''

    if not isinstance(cols_to_bin, (list, tuple, set, pd.core.indexes.base.Index)):
        cols_to_bin = [cols_to_bin]

    bin_cols = []
    
    df_copy = df.copy()
    
    for col in cols_to_bin:
        # print(col)
        num_values =  len(df[col].unique())
        if num_values == 1:
            print(f'WARNING: Column {col} contains a single value only and will be ignored')
            continue
        if isinstance(bins, int) and num_values < bins:
            print(f'WARNING: The number of bins will be reduced to {num_values}, which is the number of unique values for column {col}.')
            df_copy[col + ' bins'] = df[col]
        else:     
            df_copy[col + ' bins'] = pd.qcut(df[col], bins, duplicates='drop') if quantiles else pd.cut(df[col], bins=bins)
        bin_cols.append(col + ' bins')

    df_stat_list = []
    for col in bin_cols:
        df_stat_list.append(df_copy.groupby(col).agg(['mean', 'std', 'skew']))
    
    return df_stat_list


##############################################################################


def plot_bin_all_stat(df_stat, selected_cols, ncols=4):
    '''
    
    Plot data from df_stat

    Parameters
    ----------
    df_stat : DataFrame
        Statistics w.r.t. to a particular feature.
        Any element of a list returned by get_bin_stat
    selected_cols : str or list of str
       features to plot
    ncols : int, optional
        Number of columns for plots. The default is 4.

    Returns
    -------
    None.

    '''
    
    if not isinstance(selected_cols, (list, tuple, set, pd.core.indexes.base.Index)):
        selected_cols = [selected_cols]
        
    col_to_bin = df_stat.index.name.split(' ')[0]
    stats = df_stat.columns.get_level_values(1).unique()
    for stat in stats:
        l = len(selected_cols)
        nrows = l // ncols + int(bool(l % ncols))
        fig, ax = plt.subplots(nrows=nrows , ncols=ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle(col_to_bin + ' ' + stat)

        for i in range(nrows):
            for j in range(ncols):
                k = i * ncols + j
                if k < l:
                    col = selected_cols[k]
                    x = [interv.mid for interv in df_stat.index.values]
                    y = df_stat[(col, stat)]
                    ax[i,j].plot(x, y, label=col)
                    ax[i, j].legend()
                    
                    
                    
                    
##############################################################################

def stat_significance(df_stat_list, selected_cols):
    '''
    Calculate signicance of variations of statistics for different bins
    Return max(abs(mean(bin_i)-mean(b_j))/ min(std(bin_i), std(bin_j))) for
    each feature in selected_cols and for each feature to bin from 
    df_stat_list

    Parameters
    ----------
    df_stat_list : list
        Output of get_bin_stat
    selected_cols : str or list of str
        selected features 

    Returns
    -------
    df_max_diff : DataFrame
        Contains maximum difference for each pair (feature to bin, selected feature)

    '''
    if not isinstance(selected_cols, (list, tuple, set, pd.core.indexes.base.Index)):
        selected_cols = [selected_cols]
    
    if not isinstance(df_stat_list, (list)):
       df_stat_list = [df_stat_list]    
    
    df_max_diff = pd.DataFrame()
    
    for df_stat in df_stat_list:
        col_to_bin = df_stat.index.name.replace(' bins', '')
        for col in selected_cols:
            # col = df_stat[[col]].columns.get_level_values(0)[0]
            max_diff = 0
            for i, (x_mean, x_std) in enumerate(zip(df_stat[(col, 'mean')], df_stat[(col, 'std')])):
                for j, (y_mean, y_std) in enumerate(zip(df_stat[(col, 'mean')], df_stat[(col, 'std')])):
                    if j > i:
                        max_diff = max(np.abs(x_mean - y_mean) / min(x_std, y_std), max_diff)

            # df_max_diff.loc[(col_to_bin,  'max_diff'), col] = max_diff
            df_max_diff.loc[col_to_bin, col] = max_diff
            
    return df_max_diff


##############################################################################

def get_bin_stat_multidim(df, cols_to_bin, bins, quantiles=True):
    '''
    
    Bin every feature from cols_to_bin in df  into bins
    then calculate mean, std, and skew for all 
    features in df w.r.t. all possible combinations of that bins.
    Return a DataFrame with the corresponding statistics.

    Parameters
    ----------
    df : DataFrame
        Features
    cols_to_bin : str or list of str 
       Features to bin
    bins : int or array
        (Number of) bins. See pd.cut for more info.
    quantiles: boolean
        If True use pd.qcut for quantile-based discretization,
        otherwise use pd.cut to bin values. 
        The default is True.

    Returns
    -------
    df_bins_stat : DataFrame
        DataFrames containing statistics w.r.t. to the combinatorial product 
        of all features from cols_to_bin

    '''

    if not isinstance(cols_to_bin, (list, tuple, set, pd.core.indexes.base.Index)):
        cols_to_bin = [cols_to_bin]

    bin_cols = []
    
    df_copy = df.copy()

    for col in cols_to_bin:
        num_values =  len(df[col].unique())
        if num_values == 1:
            print(f'WARNING: Column {col} contains a single value only and will be ignored')
            continue
        if isinstance(bins, int) and num_values < bins:
            print(f'WARNING: The number of bins will be reduced to {num_values}, which is the number of unique values for column {col}.')
            df_copy[col + ' bins'] = df[col]
        else:     
            df_copy[col + ' bins'] = pd.qcut(df[col], bins, duplicates='drop') if quantiles else pd.cut(df[col], bins=bins)
        bin_cols.append(col + ' bins')

    df_stat = df_copy.groupby(bin_cols).agg(['mean', 'std', 'skew'])
    
    if any(df_stat.isnull().sum()):
        print('WARNING: Some cells contain NaN values. Reduce the number of bins to avoid it.')
    
    return df_stat


##############################################################################


def stat_significance_multidim(df_stat, selected_cols):
    '''
    Calculate signicance of variations of statistics for different combinations 
    of bins.
    Return max(abs(mean(bin_i)-mean(b_j))/ min(std(bin_i), std(bin_j))) for
    each feature in selected_cols and for each combinations of bins from 
    df_stat

    Parameters
    ----------
    df_stat : DataFrame
        Output of get_bin_stat_multidim
    selected_cols : str or list of str
        selected features 

    Returns
    -------
    df_max_diff : DataFrame
        Contains maximum difference for each feature from selected feature

    '''
    if not isinstance(selected_cols, (list, tuple, set, pd.core.indexes.base.Index)):
        selected_cols = [selected_cols]
    
    
    
    cols_to_bin = tuple(name.replace(' bins', '') for name in df_stat.index.names)
    index = pd.MultiIndex.from_tuples([cols_to_bin])
    df_max_diff = pd.DataFrame(index=index, columns=selected_cols)
    for col in selected_cols:
        max_diff = 0
        for i, (x_mean, x_std) in enumerate(zip(df_stat[(col, 'mean')], df_stat[(col, 'std')])):
            for j, (y_mean, y_std) in enumerate(zip(df_stat[(col, 'mean')], df_stat[(col, 'std')])):
                if j > i:
                    max_diff = max(np.abs(x_mean - y_mean) / min(x_std, y_std), max_diff)
                    
        df_max_diff.loc[index, col] = max_diff

    return df_max_diff


##############################################################################

def mutual_stat_significance(df, cols_to_bin, bins, selected_cols, quantiles=True, verbose=False):
    '''
    
    Bin every feature from cols_to_bin in df  into bins
    then calculate mean and std for all features in df w.r.t. that bins 
    or their combinatorial products.
   
 
    Using this data calculate signicance of variations of statistics for different bins
    Return max(abs(mean(bin_i)-mean(b_j))/ min(std(bin_i), std(bin_j))) for
    each feature in selected_cols and for each bin or combinations of bins 
    obtained in the previous step.
    
  

    Parameters
    ----------
    df : DataFrame
        Features
    cols_to_bin : str or tuple or list of strs or tuples
        Features to bin.
        For features in a tuple multidimensional statistics is calculated        
    bins : int or array
        (Number of) bins. See pd.cut for more info.    
    selected_cols : str or list of str
        selected features     
    quantiles: boolean
        If True use pd.qcut for quantile-based discretization,
        otherwise use pd.cut to bin values. 
        The default is True.
    verbose: boolean
        If True print out index of col to show the progress.
        The default value is False.

     Returns
    -------
    df_max_diff : DataFrame
    Contains maximum difference for each pair (feature or combination of features to bin, selected feature)

    '''
    if not isinstance(cols_to_bin, (list, set, pd.core.indexes.base.Index)):
        cols_to_bin = [cols_to_bin]

    if not isinstance(selected_cols, (list, tuple, set, pd.core.indexes.base.Index)):
        selected_cols = [selected_cols]
    
    df_max_diff_all = pd.DataFrame()
    for i, col in enumerate(cols_to_bin):
         if verbose:
            print(print(f'{i + 1} out of {len(cols_to_bin)} columns'))
         if not isinstance(col, (tuple)):
            # print('Not tuple', col)    
            df_stat_list = get_bin_stat(df, col, bins, quantiles)
            df_max_diff = stat_significance(df_stat_list, selected_cols)
         else:
            # print('Tuple', col)
            df_stat = get_bin_stat_multidim(df, col, bins, quantiles)
            df_max_diff = stat_significance_multidim(df_stat, selected_cols)
            
         df_max_diff_all = pd.concat([df_max_diff_all, df_max_diff])    
   
    return df_max_diff_all     


##############################################################################

def conditional_correlation_stat(df, cols_pairs, bins, selected_cols, quantiles=True, verbose=False):
    '''
    
    For a pair of two features (f1, f2) from cols_pairs and a target from 
    selected_cols calculates correlation between f2 and the target for different
    bins for f1. Return the maximum absolute value of correlations over all bins
    and detailed info for all bins.
    

    Parameters
    ----------
    df : DataFrame
        Features
    cols_pairs : tuple or list of tuples
       Tuple (f1, f2) as descibed above.  
       f1 is a feature to bin.
       f2 is a feture to calculate correlations.
    bins : int or array
        (Number of) bins. See pd.cut for more info.    
    selected_cols : str or list of str
        Selected features to calculate correlations.  
    quantiles: boolean
        If True use pd.qcut for quantile-based discretization,
        otherwise use pd.cut to bin values. 
        The default is True.
    verbose: boolean
        If True print out index of col to show the progress.
        The default value is False.    
    Returns
    -------
    DataFrame, dictionary
    DataFrame contains maximal abs(correlation) for each pair (f2, target)
    dictionary {(f1, f2): Series(index = bins of f1, value = correlation of f2 with target)}
    

    '''
    
    if isinstance(cols_pairs, (tuple)):
        cols_pairs = [cols_pairs]

    if not isinstance(selected_cols, (list, tuple, set, pd.core.indexes.base.Index)):
        selected_cols = [selected_cols]


    
    index = pd.MultiIndex.from_tuples(cols_pairs)
    df_max_corr = pd.DataFrame(index=index, columns=selected_cols)  # maximum correlation
    corr_info = {}  # detailed information for all bins
    
    for k, target in enumerate(selected_cols):
        print(f'{k + 1} out of {len(selected_cols)} targets')
        for i, (col, col2) in  enumerate(cols_pairs):
            if verbose:
                print(print(f'{i + 1} out of {len(cols_pairs)}  pairs of columns'))
            if col2 == target:
                 print(f'WARNING: Feature {col2} coincides with the traget and will be ignored')
                 continue
            df_copy = df[list(set([col, col2, target]))].copy()
            num_values =  len(df[col].unique())
            if num_values == 1:
                print(f'WARNING: Column {col} contains a single value only and will be ignored')
                continue
            if isinstance(bins, int) and num_values < bins:
                print(f'WARNING: The number of bins will be reduced to {num_values}, which is the number of unique values for column {col}.')
                df_copy[col + ' bins'] = df[col]
            else:     
                df_copy[col + ' bins'] = pd.qcut(df[col], bins, duplicates='drop') if quantiles else pd.cut(df[col], bins=bins)
           
            # print('df=',  df_copy,'\n col=',  col) 
            
            df_temp = df_copy.groupby([col + ' bins'])[col2, target].corr()[col2]
            df_max_corr.loc[(col, col2), target] =  df_temp[df_temp.index.get_level_values(1) == target].abs().max()
            corr_info[(col, col2)] = df_temp[df_temp.index.get_level_values(1) == target].droplevel(1).rename(f'{target} corr with {col2}')
            # print(f'  {i + 1} out of {len(cols_pairs)} pairs')
    if df_max_corr.isnull().sum().any():
        print('WARNING: Some cells contain NaN values. Reduce the number of bins to avoid it.')
    
    return df_max_corr, corr_info 


##############################################################################
##############################################################################

def feature_remove_performance(X_input, y, y_cont, model, cols, n_splits, 
                                exclude_columns=None, delta=0, num_days=1, verbose=0,
                                position_size=None):
    
    if exclude_columns is None:
        exclude_columns = []
    
    if not isinstance(exclude_columns, (list, tuple, set)):
        exclude_columns = [exclude_columns] 
    
    # l = train_len + test_len
    # df = df_input[:l].copy()
    # df = df_input.copy()
    X = X_input.copy()
    
    # all_cols = X.columns.values
    # label_columns =  get_label_columns(target_sym, num_days)
    performance_list = []
    
    # print(cols)
    # print(df.columns)
    for col in cols:
        print(col)
        # new_cols = np.delete(all_cols, np.where(all_cols==col)) 
        # print(new_cols)
        
       
        selected_columns = [c for c in X.columns if c not in exclude_columns + [col]]
      
        
        X_selected = X[selected_columns]
        
    
        # position_size = None if position_size_col is None else  X_train[:-num_days][position_size_col]  
           
        profit, y_test_cont_total, pred_test_total, test_ind,\
        mean_profit, num_positions, sharpe, sortino = \
                \
                cross_validation_performance(X_selected, y, y_cont, model, n_splits, delta, num_days, filename=None, 
                                  verbose=0, position_size=position_size)
           
                                                 
       
        performance_list.append({'mean_profit': mean_profit, 'sharpe': sharpe, 'sortino': sortino})
       
    return performance_list
       
#############################################################################       

def greedy_feature_selection(X, y, y_cont, model, cols, n_splits,
                              delta=0, num_days=1, reverse=True, select_metric='sortino', 
                              verbose=0, feature_rank=False,  position_size=None, n_features_penalty=0):
    
  
        
    cols = list(cols)
    cols_copy = cols.copy()
    
    
    
    profit, y_test_cont_total, pred_test_total, test_ind,\
    mean_profit, num_positions, sharpe, sortino = \
                \
                cross_validation_performance(X, y, y_cont, model, n_splits, delta, num_days, filename=None, 
                                            verbose=0, position_size=position_size)
   
     
    best_performance = {'mean_profit': mean_profit, 'sharpe': sharpe, 'sortino': sortino}[select_metric]
    if verbose > -1:
        print(f'Best performance before loop: {select_metric}  {best_performance}')
        # bp = {'mean_profit': mean_profit, 'sharpe': sharpe, 'sortino': sortino}
        # print(f'Best performance before loop: {bp}')
        
    
    exclude_columns = []
    i = 1 
    while cols:
        if verbose > 0:
            print(f'Loop over features, run {i}')
            
            
        performance_list = \
            feature_remove_performance(X, y, y_cont, model, cols, n_splits,  
                                        exclude_columns=exclude_columns,
                                        delta=delta, num_days=num_days, verbose=verbose,
                                        position_size=position_size) 
         
        
        metric_list = [x[select_metric] for x in performance_list]
        max_performance = max(metric_list)
        if i==1:
            initial_performance_list = performance_list
            
        if max_performance >= best_performance - n_features_penalty * best_performance:
            best_performance = max_performance
            if verbose > -1:
                print(f'Current best performance: {select_metric} {best_performance}')
            max_ind =  metric_list.index(max_performance)
            removed_col = cols.pop(max_ind)
            exclude_columns.append(removed_col)
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
##############################################################################

def rolling_quantile(df_input, window, cols):
    '''
    Calculate rolling quantiles

    Parameters
    ----------
    df_input : DataFrame
        input data
    window : int
        window size 
    cols : str or list of str
        list of columns of df_input, for whcich quantiles are calculated

    Returns
    -------
    df : DataFrame
        output DataFrame containing a copy of the original one and quantiles

    '''
    
    if not isinstance(cols, (list, tuple, set, pd.core.indexes.base.Index)):
        cols = [cols]
    
    df = df_input.copy()
    df_max = df[cols].dropna().rolling(window).max()
    df_min = df[cols].dropna().rolling(window).min()
    new_cols = [str(col) + ' quantile' for col in cols]
    df[new_cols] = (df[cols] - df_min) / (df_max - df_min)
    
    return df
    

##############################################################################
##############################################################################

def abs_ema_features(df_input, span, cols):
    '''
    Calculate the exponetial moving average of the aboslute values for a set of featutes

    Parameters
    ----------
    df_input : DataFrame
        input data
    span : int
        span size 
    cols : str or list of str
        list of columns of df_input, for whcich quantiles are calculated

    Returns
    -------
    df : DataFrame
        output DataFrame containing a copy of the original one and quantiles

    '''
    
    if not isinstance(cols, (list, tuple, set, pd.core.indexes.base.Index)):
        cols = [cols]
    
    df = df_input.copy()
    
    new_cols = [str(col) + ' abs_ema' for col in cols]
    df[new_cols] = abs(df_input).ewm(span=span).mean()
    
    return df


##############################################################################
##############################################################################
# See https://github.com/AutoViML/featurewiz/blob/6b870dae8dcf4f24873eb61bb48947ceb84e189c/featurewiz/featurewiz.py#L514

# def FE_remove_variables_using_SULOV_method(df_train_auto, numvars, modeltype='Classification', target='target',
#                                 corr_limit = 0.70, verbose=2)

##############################################################################
##############################################################################




##############################################################################
##############################################################################



if __name__ == '__main__':

    # X = pd.DataFrame([[1, 2, 3, 6, 7, 1], [1, 4, 6, 3, 0, 1], [1, 0, 0, 0, 0, 0], [10, 2, 1, 1, 1, 1], [10, 4, 7, 6, 5, 4], [10, 0, 0, 0, 0, 0]])
    
    # corr1, clstrs, silh_coef_optimal = clusterKMeansBase(X, maxNumClusters=3, n_init=10, debug=True)
    # print('Result: ', corr1, clstrs, silh_coef_optimal, sep='\n')
    
    # clf = DecisionTreeClassifier(criterion='entropy', 
    #                              max_features=1, 
    #                              class_weight='balanced', 
    #                              min_weight_fraction_leaf=0)
                                 
    # clf = BaggingClassifier(base_estimator=clf, 
    #                       n_estimators=1000, 
    #                       max_features=1., 
    #                       max_samples=1., 
    #                       oob_score=False)
    
    ##############################################################################
    # TEST clustered_MDA_feature_imp()
    ##############################################################################
    X, y = getTestData(40, 5, 30, 10000, sigmaStd=.1)
    # corr0, clstrs, silh = clusterKMeansBase(X.corr(), maxNumClusters=10, n_init=10)
    clf = DecisionTreeClassifier(criterion='entropy', 
                                  max_features=1, 
                                  class_weight='balanced', 
                                  min_weight_fraction_leaf=0)
                                 
    clf = BaggingClassifier(base_estimator=clf, 
                          n_estimators=1000, 
                          max_features=1., 
                          max_samples=1., 
                          oob_score=False)
    # fit = clf.fit(X,y)
    
    
    clstrs, imp = clustered_MDA_feature_imp(X, y, clf, maxNumClusters=10, n_init=10, n_splits=10, num_real=1, random_state=None)
    

    # imp.sort_values('mean', inplace=True)
    # plt.figure(figsize=(10, 5))
    # imp['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
    # plt.title('Figure 6.6 Clustered MDA')
    # plt.show()
    
    
    ##############################################################################
    # TEST mutual_stat_significane()
    ##############################################################################
    # l = 1000
    # np.random.seed(4)
    # df = pd.DataFrame({'a': np.random.choice(range(0, 2), size=l), 'b':range(l), 'c': np.random.uniform(size=l),  'd': np.random.uniform(size=l), 'e': np.random.uniform(size=l)})
    # print(df.head(), '\n')
    # cols_to_bin = [('a', 'd'), 'a', ('d', 'e'), ('c', 'd', 'e')]
    # selected_cols = df.columns 
    # bins = 4
    # df_stat_significance = mutual_stat_significane(df, cols_to_bin, bins, selected_cols, quantiles=True)
    # print(df_stat_significance)
    
    
      ##############################################################################
    # TEST conditional_correlation_stat()
    ##############################################################################
    # l = 1000
    # np.random.seed(4)
    # x = np.random.choice(range(0, 2), size=l)
    # y = np.random.choice(range(0, 3), size=l)
    # target = np.random.uniform(size=l)
    # z =  np.random.uniform(size=l)
    # z = [a if k != 2 else b for a, b, k in zip(z, target, y)]
    # df = pd.DataFrame({'x': x,  'y': y, 'z': z,  'target': target})
    # print(df.head(), '\n')
    # cols_pairs = [('x', 'y'), ('y', 'z')]
    # selected_cols = 'target' 
    # bins = 4
    # df_conditional_correlation, corr_info = conditional_correlation_stat(df, cols_pairs, bins, selected_cols, quantiles=True) 
    
    # print(df_conditional_correlation)
    # print(corr_info)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    