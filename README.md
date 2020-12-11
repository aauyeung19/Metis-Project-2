# MovieMagic

## Objective:
Build a model to predict Domestic Gross of films released in the United States

## Methodology
Thinking about the metrics for success of a movie, I thought about the experience of the members of a production team currently working on a film.  To apply a number to that experience, I looked to the success of past movies that a crew member would have worked on.  Depending on the role of the member, I engineered features describing either the past average domestic gross or the average working budget of their prior movies.  The details vary from role to role.  For example the experience of someone working in a production role would be more tied to their experience working on average budget.  On the other side, successful actors/actresses would have their success reflected in the domestic gross of their prior movies. 
To prevent leakage of features, the model was trained on a time-series-split with varrying folds to ensure validity.  I did not want a person's future success to be a metric for predicting past success.  Data was run through Sklearn's Linear Regression, Ridge, and Lasso Models.  The statsmodel OLS was also used for determining colinearity and strength of the coefficients. 

## Data:
Movies were scraped from [BoxOfficeMojo](https://www.boxofficemojo.com)
Movies considered were movies from 2000 to 2019 released in over 1500 theaters

A [csv](https://www.kaggle.com/gsnehaa21/federal-holidays-usa-19662020) of US Federal Holidays was also used to investigate the affect of release date on the amount a film grossed.  


## Technologies:
* BeautifulSoup and Selenium webscraping
* Linear Regression 
* Lasso Regression and LassoCV 
* Ridge Regression and RidgeCV 
* Sklearn 
* Pipelines with Polynomial Features 

## Summary
Although Lasso and Ridge were used separately to train data, the linear regression held to have the highest R2 value and lowest RMSE. My end model had an R2 value of 0.79 and a RMSE of about 48 million.  My model was also better at predicting lower grossing films.  Something I noted when looking at the holdout data was that the model cannot plan for boxoffice blockbusters and underpredicts their domestic gross. 
Moving forward I would want to use a clustering algorithm to better link the categorical features of my data. 
