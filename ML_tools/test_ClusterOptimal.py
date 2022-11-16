# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 21:15:18 2021

@author: mstefanski
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, SpectralClustering
from folib.ML_tools.ClusterOptimal import ClusterOptimal

COLORS = {i: plt.cm.Paired(i) for i in range(20)}
# %%


def test_gaussian_grouping_five_blobs():
    # generate 2d classification dataset, making one feature of different scale
    X, y = make_blobs(n_samples=10000, centers=8, n_features=2, random_state=27)
    X[:, 0] = (X[:, 0] + 27) * 4.
    # scatter plot, dots colored by class value
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))

    fig, ax = plt.subplots()
    ax.set_title('Actual Grouping')
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=COLORS[key])

    clf = ClusterOptimal(debug=True)
    clf.fit(df[['x', 'y']].values)
    y_pred = clf.predict(df[['x', 'y']].values)

    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y_pred))

    fig, ax = plt.subplots()
    ax.set_title('Fitted Grouping')
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=COLORS[key])
    plt.close()


def test_kmeans_grouping_seven_blobs():
    # different tests using standard scaler and KMeans
    X, y = make_blobs(n_samples=10000, centers=5, n_features=2, random_state=2)
    X[:, 0] = (X[:, 0] + 7) * 4.
    # scatter plot, dots colored by class value
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))

    fig, ax = plt.subplots()
    ax.set_title('Actual Grouping')
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=COLORS[key])

    clf = ClusterOptimal(k_potential=list(range(2, 20)), scaler='standard',
                         cluster_object=KMeans, debug=True)
    clf.fit(df[['x', 'y']].values)
    y_pred = clf.predict(df[['x', 'y']].values)

    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y_pred))

    fig, ax = plt.subplots()
    ax.set_title('Fitted Grouping')
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=COLORS[key])
    plt.close()
