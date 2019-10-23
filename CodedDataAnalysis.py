import numpy as np 
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize 
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
import random

from scrips import chi2calc

# load the final compiled dataset
df = pd.read_csv('FinalData.csv')

# data cleaning
df.awardAmount = df.awardAmount/1000000 #convert from dollars to millions of dollars
print(df.recipientType.unique())
# turn all of the For Profits to For-profit
df.loc[(df.recipientType == 'For-Profit'), "recipientType"] = "For-profit"

# drop blank/nan values for recipient type (for now)
df = df[(df.recipientType=='For-profit') | (df.recipientType=='Non-profit')]
df['ForProf'] = 0
df.loc[(df.recipientType=='For-profit'), 'ForProf'] = 1


print(df.recipientType.value_counts())



print(df.techCat1.unique())
df['TechCatDummy'] = 0 # transportation fuels is default 
df.loc[(df.techCat1=='Distributed Generation'), 'TechCatDummy'] = 1
df.loc[(df.techCat1=='Transportation Storage'), 'TechCatDummy'] = 2
df.loc[(df.techCat1=='Storage'), 'TechCatDummy'] = 3
df.loc[(df.techCat1=='Building Efficiency'), 'TechCatDummy'] = 4
df.loc[(df.techCat1=='Resource Efficiency'), 'TechCatDummy'] = 5
df.loc[(df.techCat1=='Manufacturing Efficiency'), 'TechCatDummy'] = 6
df.loc[(df.techCat1=='Centralized Generation'), 'TechCatDummy'] = 7
df.loc[(df.techCat1=='Electrical Efficiency'), 'TechCatDummy'] = 8
df.loc[(df.techCat1=='Grid'), 'TechCatDummy'] = 9
df.loc[(df.techCat1=='Transportation Vehicles'), 'TechCatDummy'] = 10
df.loc[(df.techCat1=='Transportation Network'), 'TechCatDummy'] = 11


print(list(df.columns))

# calculate the chi2 based on open/designed outcomes 
stat, p, dof, expected = chi2calc(df, 'FinalDecision')
print('stat', stat, 'p', p, 'dof', dof, 'expected', expected)

# run some logistic regressions

# model 1
df = df[df.FinalDecision!='blank']
print(type(df.OPEN))
mdl1 = sm.MNLogit(df.FinalDecision, df.OPEN).fit()
print(mdl1.summary())

# add award amounts
exog = df[['awardAmount', 'OPEN']]
mdl2 = sm.MNLogit(df.FinalDecision, exog).fit()
print(mdl2.summary())

print(min(df.awardAmount))
print(np.mean(df.awardAmount))

# awardee type
#print(df.recipientType.head(20))

exog = df[['ForProf', 'OPEN']]
mdl3 = sm.MNLogit(df.FinalDecision, exog).fit()
print(mdl3.summary())

# tech category # not working 
exog = df[['TechCatDummy', 'OPEN']]
mdl4 = sm.MNLogit(df.FinalDecision, exog).fit()
print(mdl4.summary())

# partners

# all vars


mod = sm.MNLogit(df.FinalDecision, df.awardAmount).fit()
print(mod.summary())


