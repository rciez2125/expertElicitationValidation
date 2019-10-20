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

# awardee type
#print(df.recipientType.head(5))
#exog = df[['recipientType', 'OPEN']]
#mdl3 = sm.MNLogit(df.FinalDecision, exog).fit()
#print(mdl3.summary())

# tech category

# partners

# all vars


mod = sm.MNLogit(df.FinalDecision, df.awardAmount).fit()
print(mod.summary())


