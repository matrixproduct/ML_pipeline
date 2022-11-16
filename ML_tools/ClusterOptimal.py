# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 08:48:38 2021

more work to do: https://www.ijcaonline.org/archives/volume181/number21/ashour-2018-ijca-917932.pdf

comparison of clustering techniques
https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py

@author: aziabak
"""


# %%
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from folib.ML_tools.MADScaler import MADScaler
# %%


def calc_distance(x1, y1, a, b, c):
    d = abs((a * x1 + b * y1 + c)) / (np.sqrt(a * a + b * b))
    return d


class ClusterOptimal(BaseEstimator, ClusterMixin):
    def __init__(self, k_potential=list(range(2, 13)) + [24, 52], scaler=None, cluster_object=None,
                 debug=False):
        """

        Parameters
        ----------
        k_potential : list of int, optional number of clusters to consider
            The default is list(range(2, 13)) + [24, 52].
        scaler : string or sklearn Scaler instance, optional
            'standard' 0 mean 1 SD scaler, 'robust' scales by quantiles, None uses MADScaler
        cluster_object : class, optional
            Which mixture or clustering object to use, GaussianMixture, DBSCAN, SpectralClustering
        debug : boolean, optional whether to show a graph of dfAIC

        """
        self.k_potential = k_potential
        if scaler is None:
            scaler = MADScaler()
        elif isinstance(scaler, str):
            if scaler == 'standard':
                scaler = StandardScaler()
            if scaler == 'robust':
                scaler = RobustScaler()
        else:
            err_msg = f'{scaler} has to be None, standard, robust or instance of a Scaler'
            assert isinstance(scaler, TransformerMixin), err_msg
        self.scaler = scaler
        if cluster_object is None:
            cluster_object = GaussianMixture
        else:
            assert issubclass(cluster_object, BaseEstimator)
        self.cluster_object = cluster_object
        self.debug = debug

    def fit(self, X, y=None):
        """

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the specified axis.
        y : ignored by api convention

        """
        X = self._validate_data(X, accept_sparse="csc", estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite="allow-nan")
        X_scl = self.scaler.fit(X).transform(X)
        dfAIC = pd.Series(dtype=np.float64)
        potential_models = {}
        for k_i in self.k_potential:
            clustering = self.cluster_object(k_i, random_state=42).fit(X_scl)
            potential_models[k_i] = clustering
            if hasattr(clustering, 'aic'):
                dfAIC.loc[k_i] = clustering.aic(X_scl)
            else:
                dfAIC.loc[k_i] = clustering.score(X_scl)

        if self.debug is True:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_title('Cost curve')
            dfAIC.plot()

        a = dfAIC.iloc[0] - dfAIC.iloc[-1]
        b = dfAIC.index[-1] - dfAIC.index[0]
        c1 = dfAIC.iloc[0] * dfAIC.index[0]
        c2 = dfAIC.iloc[-1] * dfAIC.index[-1]
        c = c1 - c2

        # find min AIC
        dfCost = dfAIC * np.nan
        for ki, vi in dfAIC.items():
            dfCost[ki] = calc_distance(ki, vi, a, b, c)

        self.k_opt_ = dfCost.index[dfCost.argmin()]
        self.k_cost_ = dfCost
        self.best_model_ = potential_models[self.k_opt_]
        self.dfAIC_ = dfAIC

        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the specified axis.

        Returns
        -------
        y_pred : array of predicted classes

        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse="csc", estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite="allow-nan")
        X_scl = self.scaler.transform(X)
        y_pred = self.best_model_.predict(X_scl)

        return y_pred
