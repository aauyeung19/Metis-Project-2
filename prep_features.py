#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:21:50 2020

@author: andrew
"""

import pickle
import pandas as pd
import numpy as np
import datetime as dt
import re

movie_df = pickle.load(open('clean_movie_df.pickle', 'rb'))

# read holidays csv
holidays = pd.read_csv('usholidays.csv')

# convert all dates to datetime/timestamp
movie_df.release = pd.to_datetime(movie_df.release)
holidays.loc[:,'Date'] = pd.to_datetime(holidays.Date)

# filter out holiday dates before the year 2000
holidays = holidays[holidays.Date >= dt.datetime(2000,1,1)].Date

#########################IF I have Time: Try to replace this with numpy array combinations
def find_closest_holiday(date):
    """
    Finds the days to the nearest US Federal Holiday
    args:
    date (timestamp)
    returns:
    nearest_hol (int) days to the nearest holiday
    """
    nearest_hol = 365
    for hol in holidays:
        d = abs((hol - date).days)
        if d < nearest_hol:
            nearest_hol = d
    return nearest_hol


movie_df['holiday_distance'] = movie_df.release.map(find_closest_holiday)
# movie_df.drop(columns='release', inplace=True)

columns_to_count = ['Director',
                    'distr',
                    'Composer',
                    'Cinematographer',
                    'Editor',
                    'Production Designer',
                    'Writer',
                    'Producer',
                    'Actors']
# Gets number of times the crewmember or distributor works on something in
# the dataframe.
# Larger n may associate with more experience
for crew in columns_to_count:
    curr_portfolio = movie_df[crew].value_counts().reset_index()
    curr_portfolio = pd.DataFrame(curr_portfolio)
    curr_portfolio = curr_portfolio.rename(columns={
        'index': crew, crew: crew + '_portfolio'})
    movie_df = movie_df.merge(curr_portfolio, left_on=crew, right_on=crew)
    # Some movies have multiple people in the same row.
    # Split them to get their individual average amounts
    curr = movie_df[['title', crew]]
    # make list to append unique title/name combination
    curr_crew = []
    for row in range(len(curr)):
        movietitle = curr.iloc[row][0]
        for name in curr.iloc[row][1].split(','):
            name = name.strip()
            curr_crew.append((movietitle, name))
    curr_crew = pd.DataFrame(curr_crew, columns=['title', crew])
    # add budget and domestic gross to each member's
    # row by matching their title.
    curr_crew = curr_crew.merge(
        movie_df[['title', 'budget', 'gross_dom']], on='title')
    # get average budget by member
    mean_curr_crew = curr_crew.groupby(crew, as_index=False).mean()
    curr_crew = curr_crew.merge(
        mean_curr_crew, on=crew, suffixes=('_', '_'+crew))
    # get average budgets by title.  (For movies with multiple roles in crew)
    curr_crew = curr_crew.groupby(
        'title', as_index=False).agg({
            'budget_'+crew: 'mean', 'gross_dom_'+crew: 'mean'})
    movie_df = movie_df.merge(curr_crew, on='title')

# get dummies for genres
movie_df.set_index('title', inplace=True)
movie_df = movie_df.merge(movie_df.genres.str.get_dummies(sep=', '), left_index=True, right_index=True, );

#get dummies for mpaa rating
movie_df = pd.get_dummies(movie_df, columns=['mpaa'], drop_first=False)
    
def save_pickle():
    with open('processed_movie_df.pickle', 'wb') as to_write:
        pickle.dump(prepped, to_write)    