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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn import metrics
from sklearn.cluster import MeanShift, estimate_bandwidth
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import sys

movie_df = pickle.load(open('src/processed_movie_df.pickle', 'rb'))
columns_to_drop = ['link_stub', 'desc', 'distr', 'opening', 'release',
                   'runtime', 'genres', 'gross_dom', 'gross_inter',
                   'gross_world', 'Director', 'Writer', 'Producer', 'Composer',
                   'Cinematographer', 'Editor', 'Production Designer',
                   'Actors', 'Action', 'Adventure', 'Music', 'Thriller',
                   'mpaa_R', 'Producer_max_dom',
                   'distr_max_dom', 'Actors_max_dom', 'Animation', 'Crime',
                   'mpaa_G', 'Horror','holiday_distance',
                   'Editor_max_budg', 'Production Designer_mean_budg',
                   'Biography', 'Musical', 'average_row_budget',
                   'Composer_max_budg', 'mpaa_PG-13', 'budget_cat_low',
                   'distr_max_budg', 'runtime_cat_long', 'Fantasy',
                   'Cinematographer_max_budg', 'Writer_mean_budg',
                   'Documentary', 'Mystery', 'seasons_spring', 'Comedy',
                   'Drama', 'Family', 'History', 'Romance', 'Sci-Fi', 'Sport',
                   'War', 'Western', 'Director_mean_dom', 'Composer_max_dom',
                   'Cinematographer_max_dom', 'Production Designer_mean_dom',
                   'Producer_max_budg', 'Director_max_budg', 'Editor_max_dom',
                   'Writer_max_budg', 'Writer_max_dom', 'Writer_mean_dom',
                   'seasons_summer', 'theaters']

"""
'War',
       'Crime', 'Drama', 'Sci-Fi', 'mpaa_PG-13', 'seasons_summer',
       'budget_Composer', 'Animation', 'Romance', 'seasons_winter',
       'runtime_cat_long', 'average_row_budget']
"""
movie_df.sort_values(by='release', inplace=True)
mask = movie_df  # "movie_df.gross_dom<6e8"
X = movie_df.drop(columns=columns_to_drop)
y = movie_df.gross_dom

tss = TimeSeriesSplit(n_splits=3)
# split train test split 
for train_index, test_index in tss.split(X):
    X_train, X_holdout = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_holdout = y.iloc[train_index], y.iloc[test_index]

tss2 = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tss2.split(X_train):
    X_train2, X_test = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
    y_train2, y_test = y_train.iloc[train_index], y_train.iloc[test_index]


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
    # kf = KFold(n_splits=n, shuffle=True, random_state=rs)
    tss2 = TimeSeriesSplit(n_splits=n)
    result = cross_validate(lr, X, y, cv=tss2, scoring=scoring_metrics,
                            return_train_score=True, return_estimator=True)
    return pd.DataFrame(result), result['estimator'][-1]

def get_scores(y_true, y_pred):
    MAE = metrics.mean_absolute_error(y_true, y_pred)
    MSE = metrics.mean_squared_error(y_true, y_pred)
    RMSE = np.sqrt(MSE)
    print('MAE: {}'.format(MAE))
    print('MSE: {}'.format(MSE))
    print('RMSE: {}'.format(RMSE))
    
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
                                                        test_size=0.33)
    model = sm.OLS(y_train, X_train)
    fit = model.fit()
    return fit.summary()
fit_sum = sm_LR(X_train, y_train)
fit_sum

def Kfold_Ridge(X, y, n, rs=None):
    # scale data first
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train.values)
    X_te = scaler.transform(X_test.values)
    alphavec = 10**np.linspace(-2, 2, 200)
    kf = KFold(n_splits=n, shuffle=True, random_state=rs)
    ridge_model = RidgeCV(alphas = alphavec, cv=kf)
    ridge_model.fit(X_tr, y_train)
    return ridge_model

def Kfold_Lasso(X, y, n, rs=None):
    # scale data first
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X)
    # X_te = scaler.transform(X_test.values)
    alphavec = 10**np.linspace(-2,2,200)
    tss = TimeSeriesSplit(n_splits=5)
    lasso_model = LassoCV(alphas = alphavec, cv=tss, tol=0.00001, max_iter=10000)
    lasso_model.fit(X_tr, y)
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


_, lr = Kfold_LR(X_train, y_train, 5)
rdg = Kfold_Ridge(X_train,  y_train, 5)
las = Kfold_Lasso(X_train, y_train, 5)

lr_preds = lr.predict(X_holdout)
lr_resids = lr_preds - y_holdout

scaler = StandardScaler()
X_h = scaler.fit_transform(X_holdout)
rdg_preds = rdg.predict(X_h)
rdg_resids = rdg_preds - y_holdout

lasso_preds = las.predict(X_h)
lasso_resids = lasso_preds - y_holdout

print('Linear Regression Scores')
get_scores(y_holdout, lr_preds)
print('\nRidgeCV Scores')
get_scores(y_holdout, rdg_preds)
print('\nLassoCV Scores')
get_scores(y_holdout, lasso_preds)

plt.figure()
sns.distplot(lr_resids)
plt.xlabel('Residuals')
plt.title('Linear Regression Residuals')
plt.xlim(-7e8, 2e8)
plt.ylim(0,1.8e-8)
x0, x1 = -5e8, -2.5e8
y0 = 0.2e-8
h = 0.1e-8
plt.plot([x0, x0, x1, x1],[y0-h, y0, y0, y0-h])
plt.text((x0+x1)/2, y0+0.5e-9, 'Blockbuster Movies', ha='center' )

plt.figure()
ax = sns.distplot(rdg_resids)
plt.xlabel('Residuals')
plt.title('Ridge Residuals')
plt.xlim(-7e8, 2e8)
plt.ylim(0,1.8e-8)
ax.annotate('Under Predicts',
            xy=(-1.5e8, 0.1e-8),
            xytext=(-5e8, 1e-8),
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.figure()
ax = sns.distplot(lasso_resids)
plt.xlabel('Residuals')
plt.title('Lasso Residuals')
plt.xlim(-7e8, 2e8)
plt.ylim(0,1.8e-8)
ax.annotate('Higher Bump in Tail',
            xy=(-1.5e8, 0.1e-8),
            xytext=(-5e8, 1e-8),
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.figure()
plt.title('Linear Regression Residuals')
plt.ylabel('Residuals')
plt.xlabel('Predictions')
sns.scatterplot(x=lr_preds, y=lr_resids)

plt.figure()
plt.xlabel('True Domestic Gross')
plt.ylabel('Predicted Domestic Gross')
plt.plot([0,8e8],[0,8e8], color='r')
plt.scatter(x=y_holdout, y=lr_preds, linewidths=(0.5), edgecolors=('black'))
plt.title('Fitted Relationship')

