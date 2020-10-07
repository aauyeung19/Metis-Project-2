#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleaning All Data scraped from boxofficemojo.com

Created on Mon Oct  5 10:32:50 2020

@author: andrew
"""
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn import metrics
from sklearn.cluster import MeanShift, estimate_bandwidth
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats

movie_df = pickle.load(open('processed_movie_df.pickle', 'rb'))
columns_to_drop=['link_stub', 'release','desc', 'distr', 'opening', 'runtime',
       'genres', 'gross_dom', 'gross_inter', 'gross_world',
       'Director', 'Writer', 'Producer', 'Composer', 'Cinematographer',
       'Editor', 'Production Designer', 'Actors', 'Action', 'Adventure',
       'Music', 'Thriller', 'mpaa_R'] 

""" , 'War',
       'Crime', 'Drama', 'Sci-Fi', 'mpaa_PG-13', 'seasons_summer',
       'budget_Composer', 'Animation', 'Romance', 'seasons_winter',
       'runtime_cat_long', 'average_row_budget']
"""
mask = movie_df # "movie_df.gross_dom<6e8"
X = movie_df.drop(columns=columns_to_drop)
y = movie_df.gross_dom

def Kfold_LR(X, y, n, rs=None):
    """
    Performs a Linear Regression with a Kfold CV

    Parameters
    ----------
    X : (DataFrame)
        Features.
    y : (Series)
        Target Variable.
    n : (int)
        Number of folds for Kfold.
    rs : (int)
        Random State.

    Returns
    -------
    (DataFrame)
        cross_validate Dataframe.

    """
    scoring_metrics = ['neg_mean_absolute_error',
                       'neg_mean_squared_error',
                       'neg_root_mean_squared_error',
                       'r2']
    lr = LinearRegression()
    kf = KFold(n_splits=n, shuffle=True, random_state=rs)
    result = cross_validate(lr, X, y, cv=kf, scoring=scoring_metrics,
                            return_train_score=True)
    return pd.DataFrame(result)


def sm_LR(X, y):
    '''
    Linear Regression using Stats Models

    Parameters
    ----------
    X : (DataFrame)
        Features.
    y : (Series)
        Target Variable.

    Returns
    -------
    fit
        Retrusn fit model.

    '''
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        random_state=42)
    model = sm.OLS(y_train, X_train)
    fit = model.fit()
    return fit

def Kfold_Ridge(X, y, n, rs):
    # scale data first
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train.values)
    X_te = scaler.transform(X_test.values)
    alphavec = 10**np.linspace(-2, 2, 200)
    kf = KFold(n_splits=n, shuffle=True, random_state=rs)
    ridge_model = RidgeCV(alphas = alphavec, cv=kf)
    ridge_model.fit(X_tr, y_train)
    return ridge_model

def Kfold_Lasso(X, y, n, rs):
    # scale data first
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train.values)
    X_te = scaler.transform(X_test.values)
    alphavec = 10**np.linspace(-2,2,200)
    kf = KFold(n_splits=n, shuffle=True, random_state=rs)
    lasso_model = LassoCV(alphas = alphavec, cv=5)
    lasso_model.fit(X_tr, y_train)
    return lasso_model
    

def show_normal_qq(resid):
    '''
    Show QQ plot based on residuals

    Parameters
    ----------
    resid : (array)
        Array with residuals.

    Returns
    -------
    None.

    '''
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot")
    plt.show()

