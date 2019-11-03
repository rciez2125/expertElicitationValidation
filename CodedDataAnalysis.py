import numpy as np 
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize 
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
import random
#import rpy2
#from rpy2.robjects.packages import importr
#base = importr('base')
#utils = importr('utils')
#vcov = importr('vcov')

from scrips import chi2calc

# load the final compiled dataset
df = pd.read_csv('FinalData.csv')

# do some data cleaning
def cleanData(df):
	df = df.drop(['Unnamed: 0.1'], axis=1)
	df.loc[(df.FinalDecision == 'Persist '), "FinalDecision"] = "Persist"

	df.awardAmount = df.awardAmount/1000000 #convert from dollars to millions of dollars
	#print(df.recipientType.unique())

	# turn all of the For Profits to For-profit
	df.loc[(df.recipientType == 'For-Profit'), "recipientType"] = "For-profit"
	
	# year codes 
	df['endYr'] = ""
	df['startYr'] = ""
	df['yrGrp'] = ""
	df['early'] = 0
	df['middle'] = 0
	df['late'] = 0
	df['dum09'] = 0
	df['dum10'] = 0
	df['dum11'] = 0
	df['dum12'] = 0
	df['dum13'] = 0
	df['dum14'] = 0
	df['dum15'] = 0
	df['dum16'] = 0
	df['dum17'] = 0
	df['dum18'] = 0
	for n in range(df.shape[0]):
		s = df.endDate[n]
		df.endYr[n] = float(s[:4])
		s = df.startDate[n]
		df.startYr[n] = float(s[:4])

		if df.startYr[n] < float(2012):
			df.yrGrp[n] = 0
			df.early[n] = 1
		elif df.startYr[n] < float(2015):
			df.yrGrp[n] = 1
			df.middle[n] = 1
		else:
			df.yrGrp[n] = 2
			df.late[n] = 1

		if df.startYr[n] < float(2010):
			df.dum09[n] = 1
		elif df.startYr[n] < float(2011):
			df.dum10[n] = 1
		elif df.startYr[n] < float(2012):
			df.dum11[n] = 1
		elif df.startYr[n] < float(2013):
			df.dum12[n] = 1
		elif df.startYr[n] < float(2014):
			df.dum13[n] = 1
		elif df.startYr[n] < float(2015):
			df.dum14[n] = 1
		elif df.startYr[n] < float(2016):
			df.dum15[n] = 1
		elif df.startYr[n] < float(2017):
			df.dum16[n] = 1
		elif df.startYr[n] < float(2018):
			df.dum17[n] = 1
		else:
			df.dum18[n] = 1

	df.endYr = df.endYr - 2010
	df.startYr = df.startYr - 2009

	# drop blank/nan values for recipient type (for now)
	df = df[(df.recipientType=='For-profit') | (df.recipientType=='Non-profit')]
	df['ForProf'] = 0
	df.loc[(df.recipientType=='For-profit'), 'ForProf'] = 1
	df = df[(df.awardAmount!=0)]

	df.reset_index(drop = True)

	print(df.recipientType.value_counts())

	df['TC_TF'] = 0 # transportation Fuels
	df['TC_DG'] = 0 # distributed generation
	df['TC_TS'] = 0 # transportation storage
	df['TC_SS'] = 0 # stationary storage
	df['TC_BE'] = 0 # building efficiency 
	df['TC_RE'] = 0 # resource efficiency 
	df['TC_ME'] = 0 # manufacturing efficiency 
	df['TC_CG'] = 0 # centralized generation
	df['TC_EE'] = 0 # electrical efficiency 
	df['TC_GR'] = 0 # grid  
	df['TC_TV'] = 0 # transportation vehicles
	df['TC_TN'] = 0 # transportation network
	df['TC_OT'] = 0 # fewer than 10 projects in a category #Transportation Network, Transportation Vehicles, and Centralized Generation

	df.loc[(df.techCat1=='Transportation Fuels'), 'TC_TF'] = 1
	df.loc[(df.techCat1=='Distributed Generation'), 'TC_DG'] = 1
	df.loc[(df.techCat1=='Transportation Storage'), 'TC_TS'] = 1
	df.loc[(df.techCat1=='Storage'), 'TC_SS'] = 1
	df.loc[(df.techCat1=='Building Efficiency'), 'TC_BE'] = 1
	df.loc[(df.techCat1=='Resource Efficiency'), 'TC_RE'] = 1
	df.loc[(df.techCat1=='Manufacturing Efficiency'), 'TC_ME'] = 1
	df.loc[(df.techCat1=='Centralized Generation'), 'TC_CG'] = 1
	df.loc[(df.techCat1=='Electrical Efficiency'), 'TC_EE'] = 1
	df.loc[(df.techCat1=='Grid'), 'TC_GR'] = 1
	df.loc[(df.techCat1=='Transportation Vehicles'), 'TC_TV'] = 1
	df.loc[(df.techCat1=='Transportation Network'), 'TC_TN'] = 1

	df.loc[(df.techCat1=='Centralized Generation'), 'TC_OT'] = 1
	df.loc[(df.techCat1=='Transportation Vehicles'), 'TC_OT'] = 1
	df.loc[(df.techCat1=='Transportation Network'), 'TC_OT'] = 1

	df.to_csv('cleanedFinalData.csv')
	return(df)
	

df = cleanData(df)

df.awardAmount = np.log(df.awardAmount)


# calculate the chi2 based on open/designed outcomes 
stat, p, dof, expected = chi2calc(df, 'FinalDecision')
print('stat', stat, 'p', p, 'dof', dof, 'expected', expected)

def makeTables(): #makes a csv version of the odds ratio 
	print('hello world')

# define some regression models 

def runMod1(pooling): #just open
	if pooling == 'total':
		exog = df[['OPEN']] 
	elif pooling == 'none':
		exog = df[['OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17']]
	elif pooling == 'fakePartial':
		exog = df[['OPEN', 'early', 'middle', 'late']]
	else:
		print('hello world') #figure out if true partial pooling is a thing for multinomial logit models, does it really mean anything? 

	mdl1 = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 1000, full_output = True)
	print(mdl1.summary())
	print(np.exp(mdl1.params))

	mod1mg = mdl1.get_margeff(at='overall')
	print(mod1mg.summary())
	print(mod1mg.summary_frame())
	return(mod1mg.summary_frame())

def runMod2(pooling): # add award amounts
	if pooling == 'total':
		exog = df[['awardAmount','OPEN']] 
	elif pooling == 'none':
		exog = df[['awardAmount', 'OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17']]
	elif pooling == 'fakePartial':
		exog = df[['awardAmount', 'OPEN', 'early', 'middle']]
	else:
		print('hello world')

	mdl2 = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit()
	print(mdl2.summary())
	mod2mg = mdl2.get_margeff()
	print(mod2mg.summary())
	print(mod2mg.summary_frame())
	return(mod2mg.summary_frame())

def runMod3(pooling): # for profit
	if pooling == 'total':
		exog = df[['ForProf', 'OPEN']] 
	elif pooling == 'none':
		exog = df[['ForProf', 'OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17']]
	elif pooling == 'fakePartial':
		exog = df[['ForProf', 'OPEN', 'early', 'middle']]
	else:
		print('hello world')

	mdl3 = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit()
	print(mdl3.summary())
	mod3mg = mdl3.get_margeff()
	print(mod3mg.summary())

def runMod4(pooling): # tech category, relative to storage baseline 
	if pooling == 'total':
		exog = df[['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'OPEN']]
	elif pooling == 'none':
		exog = df[['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17']]
	elif pooling == 'fakePartial':
		exog = df[['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'OPEN', 'early', 'middle']]
	else:
		print('hello world')

	mdl4 = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit()
	print(mdl4.summary())
	mod4mg = mdl4.get_margeff()
	print(mod4mg.summary())

def runMod5(pooling): # partners
	if pooling == 'total':
		exog = df[['OPEN']]
	elif pooling == 'none':
		exog = df[['OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17']]
	elif pooling == 'fakePartial':
		exog = df[['OPEN', 'early', 'middle']]
	else:
		print('hello world')

	print('hello world')

def runMod6(pooling): # everything
	# add partner info 
	if pooling == 'total':
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'OPEN']]
	elif pooling == 'none':
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17']]
	elif pooling == 'fakePartial':
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'OPEN', 'early', 'middle']]
	else:
		print('hello world')

	# all vars model 6
	mdl6 = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit()
	print(mdl6.summary())
	mod6mg = mdl6.get_margeff()
	print(mod6mg.summary())

def runMod7(pooling): # everything w/o open
	# add partner info 
	if pooling == 'total':
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT']]
	elif pooling == 'none':
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17']]
	elif pooling == 'fakePartial':
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'early', 'middle']]
	else:
		print('hello world')

	mdl7 = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit()
	print(mdl7.summary())
	mod7mg = mdl7.get_margeff()
	print(mod7mg.summary())

#runMod1('total')
runMod1('fakePartial')
#runMod1('none')

#x1 = runMod2('total')
#x2 = runMod2('none')
#x3 = runMod2('fakePartial')

#print('perish')
#print(x1['dy/dx'][('FinalDecision=Perish', 'awardAmount')], x1['Pr(>|z|)'][('FinalDecision=Perish', 'awardAmount')])
#print(x2['dy/dx'][('FinalDecision=Perish', 'awardAmount')],  x2['Pr(>|z|)'][('FinalDecision=Perish', 'awardAmount')])
#print(x3['dy/dx'][('FinalDecision=Perish', 'awardAmount')],  x3['Pr(>|z|)'][('FinalDecision=Perish', 'awardAmount')])

#print('persist')
#print(x1['dy/dx'][('FinalDecision=Persist', 'awardAmount')], x1['Pr(>|z|)'][('FinalDecision=Persist', 'awardAmount')])
#print(x2['dy/dx'][('FinalDecision=Persist', 'awardAmount')], x2['Pr(>|z|)'][('FinalDecision=Persist', 'awardAmount')])
#print(x3['dy/dx'][('FinalDecision=Persist', 'awardAmount')], x3['Pr(>|z|)'][('FinalDecision=Persist', 'awardAmount')])

#print('pivot')
#print(x1['dy/dx'][('FinalDecision=Pivot', 'awardAmount')], x1['Pr(>|z|)'][('FinalDecision=Pivot', 'awardAmount')])
#print(x2['dy/dx'][('FinalDecision=Pivot', 'awardAmount')], x2['Pr(>|z|)'][('FinalDecision=Pivot', 'awardAmount')])
#print(x3['dy/dx'][('FinalDecision=Pivot', 'awardAmount')], x3['Pr(>|z|)'][('FinalDecision=Pivot', 'awardAmount')])


print('perish')
#print(x1['dy/dx'][('FinalDecision=Perish', 'OPEN')], x1['Pr(>|z|)'][('FinalDecision=Perish', 'OPEN')])
#print(x2['dy/dx'][('FinalDecision=Perish', 'OPEN')],  x2['Pr(>|z|)'][('FinalDecision=Perish', 'OPEN')])
#print(x3['dy/dx'][('FinalDecision=Perish', 'OPEN')],  x3['Pr(>|z|)'][('FinalDecision=Perish', 'OPEN')])

print('persist')
#print(x1['dy/dx'][('FinalDecision=Persist', 'OPEN')], x1['Pr(>|z|)'][('FinalDecision=Persist', 'OPEN')])
#print(x2['dy/dx'][('FinalDecision=Persist', 'OPEN')], x2['Pr(>|z|)'][('FinalDecision=Persist', 'OPEN')])
#print(x3['dy/dx'][('FinalDecision=Persist', 'OPEN')], x3['Pr(>|z|)'][('FinalDecision=Persist', 'OPEN')])

print('pivot')
#print(x1['dy/dx'][('FinalDecision=Pivot', 'OPEN')], x1['Pr(>|z|)'][('FinalDecision=Pivot', 'OPEN')])
#print(x2['dy/dx'][('FinalDecision=Pivot', 'OPEN')], x2['Pr(>|z|)'][('FinalDecision=Pivot', 'OPEN')])
#print(x3['dy/dx'][('FinalDecision=Pivot', 'OPEN')], x3['Pr(>|z|)'][('FinalDecision=Pivot', 'OPEN')])

# which pooling version? 
#runMod3('total')
#runMod4('total') # this works, is better than null 
#runMod5('total') # still need partner 
#runMod6('total') # still need partner, but works better than null 
#runMod7('total') # no open, still need partner  


#runMod3('none')
#runMod4('none')
#runMod5('none')
#runMod6('none')
#runMod7('none')
