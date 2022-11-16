# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:00:05 2021

@author: mstefanski
"""

import numpy as np
from scipy.stats import median_abs_deviation as mad
from sklearn.base import BaseEstimator, TransformerMixin, _OneToOneFeatureMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
# %%


class MADScaler(_OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """
    Transform all of the input data into MAD of 1.0  with scale='normal'
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """

    def __init__(self, *, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        """Optimize each column to have MAD of 1.0

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the specified axis.
        y : ignored by api convention

        """
        X = self._validate_data(X, accept_sparse="csc", estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite="allow-nan")
        scale = mad(X, scale='normal')
        self.scale_ = np.array(scale).reshape(1, -1)
        return self

    def transform(self, X):
        """Scale the data according to previous fitting in self.scale_

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the specified axis.

        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=("csr", "csc"), copy=self.copy, estimator=self,
                                dtype=FLOAT_DTYPES, reset=False, force_all_finite="allow-nan")
        return X / self.scale_
