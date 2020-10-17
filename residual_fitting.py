"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from data import make_data1, make_data2
from plot import plot_boundary

from scipy.stats.stats import pearsonr


class residual_fitting(BaseEstimator, ClassifierMixin):


    def fit(self, X, y):
        """Fit a Residual fitting model using the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        
        #self : weights matrices
        #w_0 = y_average
        
        self[0] = np.mean(y)
        
        #pre-whitening of attributes
        
        mean = np.mean(X,axis=0)
        std = np.std(X,axis=0)
        
        #Substraction of the mean of the attributes and normalisation of variance
        X = (X - X.mean(axis=0).reshape(1,2))/(X.std(axis=0).reshape(1,2))
        
        residual_k = np.zeros(y.shape)

        #Defining residual at step k
        for k in range(1,X.ndim+1):
            i=0
            for y_k in y:
                #Sum of w_i*a_i(o)
                linear_sum=0.
                for j in range(1,k):
                    for l in range(0,X.shape[0]):
                        linear_sum += self[j]*X[l,j-1] 
                residual_k[i] = y_k - linear_sum - self[0]
                i+=1
            #best fit of residual with only attribute a_k
            self[k]= pearsonr(residual_k,X[:,k-1])[0]*np.std(residual_k)
       
        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """
        y = np.zeros(X.shape[0])
        
        for i in range(0,y.shape[0]):
            #adding intercept
            y[i] += self[0]
            
            #sum of w_i*x_i
            for j in range(1,self.shape[0]):
                y[i] += self[j]*X[i,j-1]
        
        return y
        

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # ====================
        # TODO your code here.
        # ====================

        pass

if __name__ == "__main__":
    from data import make_data1,make_data2
    from plot import plot_boundary, plot_boundary_extended
