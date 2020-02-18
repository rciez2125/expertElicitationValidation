import numpy as np 
import pandas as pd 
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize, rosen 
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex
from matplotlib.colors import to_rgb
import random
import math

#import rpy2
#from rpy2.robjects.packages import importr
#base = importr('base')
#utils = importr('utils')
#vcov = importr('vcov')

from scrips import chi2calc

# load the final compiled dataset
df = pd.read_csv('Data/FinalData_cleaned.csv')

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
	df['dumPersist'] = 0
	df['dumPivot'] = 0
	df['dumPerish'] = 0 
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


	# small and large for profits 
	df['ForProfSize'] = 0
	df.loc[(df.recipientType=='For-profit'), 'ForProfSize'] = 1
	df.loc[(df.Size)=='Large', 'ForProfSize'] = 2	

	df['SmallForProf'] = 0
	df['LargeForProf'] = 0
	df.loc[(df.Size)=='Small', 'SmallForProf'] = 1
	df.loc[(df.Size)=='Large', 'LargeForProf'] = 1

	df.loc[(df.FinalDecision=='Perish'), 'dumPerish'] = 1
	df.loc[(df.FinalDecision=='Pivot'), 'dumPivot'] = 1
	df.loc[(df.FinalDecision=='Persist'), 'dumPersist'] = 1

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

count = df.groupby(['OPEN', 'FinalDecision']).size() 
#print(count) 
##df.awardAmount = np.log(df.awardAmount)

# calculate the chi2 based on open/designed outcomes 
stat, p, dof, expected = chi2calc(df, 'OPEN', 'FinalDecision')
#print('OPEN', 'stat', stat, 'p', p, 'dof', dof, 'expected', expected)

stat, p, dof, expected = chi2calc(df, 'recipientType', 'FinalDecision')
#print('recipientType', 'stat', stat, 'p', p, 'dof', dof, 'expected', expected)

stat, p, dof, expected = chi2calc(df, 'Partners', 'FinalDecision')
#print('Partners', 'stat', stat, 'p', p, 'dof', dof, 'expected', expected)


# nonstandard number of categories
count_series = df.groupby(['techCat1', 'FinalDecision']).size()
new_df = count_series.to_frame(name = 'breakdown').reset_index()
new_df = new_df[new_df.FinalDecision !='blank']
line = pd.DataFrame({"techCat1": "Transportation Network", "FinalDecision": 'Perish', "breakdown":0}, index=[26.5])
new_df = new_df.append(line, ignore_index=False)
new_df = new_df.sort_index().reset_index(drop=True)
stat, p, dof, expected = stats.chi2_contingency([new_df.breakdown[0:3], new_df.breakdown[3:6], new_df.breakdown[6:9], new_df.breakdown[9:12], new_df.breakdown[12:15], new_df.breakdown[15:18], new_df.breakdown[18:21], new_df.breakdown[21:24], new_df.breakdown[24:27], new_df.breakdown[27:30], new_df.breakdown[30:33], new_df.breakdown[33:36]])# ddof = 2)
print('Tech category', 'stat', stat, 'p', p, 'dof', dof, 'expected', expected)

count_series = df.groupby(['ForProfSize', 'FinalDecision']).size()
new_df = count_series.to_frame(name = 'breakdown').reset_index()
stat, p, dof, expected = stats.chi2_contingency([new_df.breakdown[0:3], new_df.breakdown[3:6], new_df.breakdown[6:9]])
print('recipientType with size', 'stat', stat, 'p', p, 'dof', dof, 'expected', expected)

count_series = df.groupby(['startYr', 'FinalDecision']).size()
new_df = count_series.to_frame(name = 'breakdown').reset_index()
line = pd.DataFrame({"startYr": 2, "FinalDecision": 'Persist', "breakdown":0}, index=[6.5])
new_df = new_df.append(line, ignore_index=False)
new_df = new_df.sort_index().reset_index(drop=True)
print(new_df)
stat, p, dof, expected = stats.chi2_contingency([new_df.breakdown[0:3], new_df.breakdown[3:6], new_df.breakdown[6:9], new_df.breakdown[9:12], new_df.breakdown[12:15], new_df.breakdown[15:18], new_df.breakdown[18:21], new_df.breakdown[21:24], new_df.breakdown[24:27]])# ddof = 2)
print('start year', 'stat', stat, 'p', p, 'dof', dof, 'expected', expected)

count_series = df.groupby(['yrGrp', 'FinalDecision']).size()
new_df = count_series.to_frame(name = 'breakdown').reset_index()
stat, p, dof, expected = stats.chi2_contingency([new_df.breakdown[0:3], new_df.breakdown[3:6], new_df.breakdown[6:9]])
print('start year - grouped', 'stat', stat, 'p', p, 'dof', dof, 'expected', expected)


# continous variable test 
exog = df[['awardAmount']] 
awardAmttest = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)# method = 'bfgs')
print(awardAmttest.summary())

lr = 2*(-498.57-(-493.11))
print('LR award amt', lr)

def makeMultipleModelStep2(time):
	if time == "full":
		exog = df[['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'SmallForProf', 'LargeForProf', 'awardAmount', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16']]
	else:
		exog = df[['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'SmallForProf', 'LargeForProf', 'awardAmount', 'early', 'middle']]

	mod = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)# method = 'bfgs')
	print(mod.summary())
	modmg = mod.get_margeff(at='overall')
	#print(modmg.summary())
	return(mod)
m1 = makeMultipleModelStep2('short')
print(m1.params)

def makeSmallerModelStep2(time):
	if time == 'full':
		exog = df[['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'ForProf', 'awardAmount', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16']]
	else:
		exog = df[['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'ForProf', 'awardAmount', 'early', 'middle']]
	mod = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)# method = 'bfgs')
	print(mod.summary())
	modmg = mod.get_margeff(at='overall')
	#print(modmg.summary())
	return(mod)
m2 = makeSmallerModelStep2('short')
print(m2.params)

x1 = m1.params
x2 = m2.params
print(type(x2))
print(x2[0]['TC_TF'])
#print(x.iloc[0].TC_TF)

x2 = x2.drop(['ForProf'])
print(x2)
print(len(x2))


print(x2.index)
x2['comparison11'] = ""
x2['comparison12'] = ""
x2['comparison21'] = ""
x2['comparison22'] = ""

for i in x2.index:
	x2['comparison11'][i] = 100*(x2[0][i]-x1[0][i])/x2[0][i]
	x2['comparison12'][i] = 100*(x1[0][i]-x2[0][i])/x1[0][i]
	x2['comparison21'][i] = 100*(x2[1][i]-x1[1][i])/x2[1][i]
	x2['comparison22'][i] = 100*(x1[1][i]-x2[1][i])/x1[1][i]
print(x1)
print(x2['comparison11'], x2['comparison12'])
print(x2['comparison21'], x2['comparison22'])

# check that the parameters didn't change that much 


# Step 4 
# adding variables back in to model 
# open 

def addOPEN():
	exog = df[['OPEN', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'SmallForProf', 'LargeForProf', 'awardAmount']]
	mod = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)# method = 'bfgs')
	print(mod.summary())
	modmg = mod.get_margeff(at='overall')
	#print(modmg.summary())
	return(mod)
x = addOPEN()

def addPartners():
	exog = df[['Partners', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'SmallForProf', 'LargeForProf', 'awardAmount']]
	mod = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)# method = 'bfgs')
	print(mod.summary())
	modmg = mod.get_margeff(at='overall')
	#print(modmg.summary())
	return(mod)
x = addPartners()



def makeAMETables(modmg): #makes a csv version of the odds ratio 
	outdf = pd.DataFrame(columns = ('Parameter', 'Perish', 'Persist', 'Pivot'))
	outdf[0]['Parameter'] = modmg(['dy/dx'][('FinalDecision=Perish', 'OPEN')])
	print(outdf)

# define some regression models 

#print(df.columns)
#s = stats.ttest_ind(df.FinalDecision, df.awardAmount)
#print(s)

def runMod1(pooling): #just open
	if pooling == 'total':
		#exog = (df[['OPEN']]) #sm.add_constant
		exog = sm.add_constant(df[['OPEN']])
	elif pooling == 'none':
		exog = (df[['OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16']])
		#exog = sm.add_constant(exog)
	elif pooling == 'fakePartial':
		exog = (df[['OPEN', 'early', 'middle', 'late']])
	else:
		print('hello world') #figure out if true partial pooling is a thing for multinomial logit models, does it really mean anything? 

	mdl1 = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)# method = 'bfgs')
	print(mdl1.summary())
	#print(np.exp(mdl1.params))

	mod1mg = mdl1.get_margeff(at='overall')
	#print(mod1mg.summary())
	#print(mod1mg.summary_frame())
	#return(mod1mg.summary_frame())
	return(mdl1.summary())

	#mdl1binary = 

#def mod1partialPooling():

x1 = math.factorial(460)/(math.factorial(sum(df.dumPersist))*math.factorial(sum(df.dumPivot))*math.factorial(sum(df.dumPerish)))
#print(x1)
def mod1Dupe(beta):
	a = np.exp(df.OPEN * beta[0])
	b = np.exp(df.OPEN * beta[1])
	c = np.exp(df.OPEN * beta[2])
	a1 = df.dumPersist * np.log(a/(a+b+c))
	b1 = df.dumPivot * np.log(b/(a+b+c))
	c1 = df.dumPerish * np.log(c/(a+b+c))
	d = a1+b1+c1
	LL = -1*sum(d)
	return LL

betastart = np.random.rand(3)-0.5
#mod1dupeRun = minimize(mod1Dupe, betastart, method = 'BFGS', options = {'maxiter':10000})
#print(mod1dupeRun)

def mod1noPoolingDupe(beta):
	a = np.exp(df.OPEN * beta[0] + df.dum09 * beta[1] + df.dum10 * beta[2] + df.dum11 * beta[3] + df.dum12*beta[4] + df.dum13*beta[5] + df.dum14*beta[6]+df.dum15*beta[7]+ df.dum16*beta[8])
	b = np.exp(df.OPEN * beta[9] + df.dum09 * beta[10] + df.dum10 * beta[11] + df.dum11 * beta[12] + df.dum12*beta[13] + df.dum13*beta[14] + df.dum14*beta[15]+df.dum15*beta[16]+ df.dum16*beta[17])
	c = np.exp(df.OPEN * beta[18] + df.dum09 * beta[19] + df.dum10 * beta[20] + df.dum11 * beta[21] + df.dum12*beta[22] + df.dum13*beta[23] + df.dum14*beta[24]+df.dum15*beta[25]+ df.dum16*beta[26])
	a1 = df.dumPersist * np.log(a/(a+b+c))
	b1 = df.dumPivot * np.log(b/(a+b+c))
	c1 = df.dumPerish * np.log(c/(a+b+c))
	d = a1+b1+c1
	LL = -1*sum(d)
	return LL

betastart = np.random.rand(27)-0.5
#mod1dupeRun = minimize(mod1noPoolingDupe, betastart, method = 'BFGS', options = {'maxiter':10000})
#print(mod1dupeRun)

def runMod2(pooling): # add award amounts
	if pooling == 'total':
		exog = df[['awardAmount','OPEN']] 
	elif pooling == 'none':
		exog = df[['awardAmount', 'OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16']]
	elif pooling == 'fakePartial':
		exog = df[['awardAmount', 'OPEN', 'early', 'middle', 'late']]
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
		exog = df[['ForProf', 'OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16']]
	elif pooling == 'fakePartial':
		exog = df[['ForProf', 'OPEN', 'early', 'middle', 'late']]
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
		exog = df[['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16']]
	elif pooling == 'fakePartial':
		exog = df[['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'OPEN', 'early', 'middle', 'late']]
	else:
		print('hello world')

	mdl4 = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit()
	print(mdl4.summary())
	mod4mg = mdl4.get_margeff()
	print(mod4mg.summary())
	return(mod4mg.summary_frame())

def runMod5(pooling): # partners
	if pooling == 'total':
		exog = df[['Partners', 'OPEN']]
	elif pooling == 'none':
		exog = df[['Partners', 'OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16']]
	elif pooling == 'fakePartial':
		exog = df[['Partners', 'OPEN', 'early', 'middle', 'late']]
	else:
		print('hello world')
	
	mdl5 = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit()
	print(mdl5.summary())
	mod5mg = mdl5.get_margeff()
	print(mod5mg.summary())
	return(mod5mg.summary_frame())

def runMod6(pooling): # everything
	# add partner info 
	if pooling == 'total':
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'Partners', 'OPEN']]
	elif pooling == 'none':
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'Partners', 'OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16']]
	elif pooling == 'fakePartial':
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'Partners', 'OPEN', 'early', 'middle', 'late']]
	else:
		print('hello world')

	# all vars model 6
	mdl6 = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit()
	print(mdl6.summary())
	mod6mg = mdl6.get_margeff()
	print(mod6mg.summary())
	return(mod6mg.summary_frame())

def runMod7(pooling): # everything w/o open
	# add partner info 
	if pooling == 'total':
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'Partners']]
	elif pooling == 'none':
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'Partners', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16']]
	elif pooling == 'fakePartial':
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'Partners', 'early', 'middle', 'late']]
	else:
		print('hello world')

	mdl7 = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit()
	print(mdl7.summary())
	mod7mg = mdl7.get_margeff()
	print(mod7mg.summary())
	return(mod7mg.summary_frame())

def calculateAlpha():
	print('hello world!')


#m1 = runMod1('total')
#print('m1', m1)
#runMod1('fakePartial')
#m1p = runMod1('none')

#print(df.dum09.value_counts())
#print(df.dum10.value_counts())
#print(df.dum11.value_counts())
#print(df.dum12.value_counts())
#print(df.dum13.value_counts())
#print(df.dum14.value_counts())
#print(df.dum15.value_counts())
#print(df.dum16.value_counts())
#print(df.dum17.value_counts())
#print(df.dum18.value_counts())

#runMod2('total')
#runMod3('total')
#y = runMod4('total')
#runMod4('fakePartial')
#runMod5('total')
#runMod5('fakePartial')
#y2 = runMod6('total')
#runMod6('fakePartial')
#runMod7('total')

#x2 = runMod2('none')
#x3 = runMod2('fakePartial')
#cmap = plt.get_cmap("tab10")

c1 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#afc7e4', '#fcbc7e', '#92e285']
def addADatapoint(barx, bary, ub, lb, label, colorinfo):
	fs = 14
	plt.bar(barx, bary, color = c1[colorinfo])
	plt.plot([barx, barx], [lb, ub], '-k')
	if bary> 0:
		plt.text(barx, ub+0.025, label, rotation = 270, horizontalalignment = 'center', verticalalignment = 'bottom', fontsize = fs)
	else:
		plt.text(barx, lb-0.025, label, rotation = 270, horizontalalignment = 'center', verticalalignment = 'top', fontsize = fs)
def makeAFig(modmg, figname):
	fs = 14
	outcomes = ('FinalDecision=Perish', 'FinalDecision=Persist', 'FinalDecision=Pivot')
	outcomeLabels = ("Perish", "Persist", "Pivot")
	plt.figure(figsize=(11.5,4.75))

	plt.subplot(position = [0.07, 0.05, 0.9, 0.9])
	m = modmg.shape[0]/3
	for n in range(len(outcomes)):
		addADatapoint(1+n*m, modmg['dy/dx'][(outcomes[n], 'OPEN')], modmg['Cont. Int. Hi.'][(outcomes[n], 'OPEN')], modmg['Conf. Int. Low'][(outcomes[n], 'OPEN')], 'OPEN', 0)
		if m==13:
			addADatapoint(2+n*m, modmg['dy/dx'][(outcomes[n], 'awardAmount')], modmg['Cont. Int. Hi.'][(outcomes[n], 'awardAmount')], modmg['Conf. Int. Low'][(outcomes[n], 'awardAmount')], 'Award Amt.', 1)
			addADatapoint(3+n*m, modmg['dy/dx'][(outcomes[n], 'ForProf')], modmg['Cont. Int. Hi.'][(outcomes[n], 'ForProf')], modmg['Conf. Int. Low'][(outcomes[n], 'ForProf')], 'For-Profit', 2)
			addADatapoint(4+n*m, modmg['dy/dx'][(outcomes[n], 'Partners')], modmg['Cont. Int. Hi.'][(outcomes[n], 'Partners')], modmg['Conf. Int. Low'][(outcomes[n], 'Partners')], 'Partners', 3)

		addADatapoint((n+1)*m-8, modmg['dy/dx'][(outcomes[n], 'TC_TF')], modmg['Cont. Int. Hi.'][(outcomes[n], 'TC_TF')], modmg['Conf. Int. Low'][(outcomes[n], 'TC_TF')], 'Trans. Fuels', 4)
		addADatapoint((n+1)*m-7, modmg['dy/dx'][(outcomes[n], 'TC_DG')], modmg['Cont. Int. Hi.'][(outcomes[n], 'TC_DG')], modmg['Conf. Int. Low'][(outcomes[n], 'TC_DG')], 'Dist. Gen.', 5)
		addADatapoint((n+1)*m-6, modmg['dy/dx'][(outcomes[n], 'TC_TS')], modmg['Cont. Int. Hi.'][(outcomes[n], 'TC_TS')], modmg['Conf. Int. Low'][(outcomes[n], 'TC_TS')], 'Trans. Storage', 6)
		addADatapoint((n+1)*m-5, modmg['dy/dx'][(outcomes[n], 'TC_BE')], modmg['Cont. Int. Hi.'][(outcomes[n], 'TC_BE')], modmg['Conf. Int. Low'][(outcomes[n], 'TC_BE')], 'Building Eff.', 7)
		addADatapoint((n+1)*m-4, modmg['dy/dx'][(outcomes[n], 'TC_RE')], modmg['Cont. Int. Hi.'][(outcomes[n], 'TC_RE')], modmg['Conf. Int. Low'][(outcomes[n], 'TC_RE')], 'Resource Eff.', 8)
		addADatapoint((n+1)*m-3, modmg['dy/dx'][(outcomes[n], 'TC_ME')], modmg['Cont. Int. Hi.'][(outcomes[n], 'TC_ME')], modmg['Conf. Int. Low'][(outcomes[n], 'TC_ME')], 'Manufact. Eff.', 9)
		addADatapoint((n+1)*m-2, modmg['dy/dx'][(outcomes[n], 'TC_EE')], modmg['Cont. Int. Hi.'][(outcomes[n], 'TC_EE')], modmg['Conf. Int. Low'][(outcomes[n], 'TC_EE')], 'Elec. Eff.', 10)
		addADatapoint((n+1)*m-1, modmg['dy/dx'][(outcomes[n], 'TC_GR')], modmg['Cont. Int. Hi.'][(outcomes[n], 'TC_GR')], modmg['Conf. Int. Low'][(outcomes[n], 'TC_GR')], 'Grid', 11)
		addADatapoint((n+1)*m, modmg['dy/dx'][(outcomes[n], 'TC_OT')], modmg['Cont. Int. Hi.'][(outcomes[n], 'TC_OT')], modmg['Conf. Int. Low'][(outcomes[n], 'TC_OT')], 'Other', 12)

		plt.plot([(n+1)*m+0.5, (n+1)*m+0.5], [-2, 2], '--k')
		plt.text((n*m)+m/2+0.5, 1.45, outcomeLabels[n], horizontalalignment = 'center', fontsize = fs)

	plt.ylabel('Average Marginal Effect', fontsize = fs)
	plt.plot([0,m*4],[0,0], '-k')
	plt.ylim(-1.4, 1.4)
	plt.xticks(np.linspace(0,40,9), ' ')
	plt.xlim(0.5, m*3+0.5)
	#plt.xticks('')
	plt.savefig(figname, dpi = 300)

#makeAFig(y, 'bargraph_TechCat.png')
#makeAFig(y2, 'bargraph_allVars.png')

def makeFrequencyBarChart(df):
	print(df.columns)
	y1=(df.FinalDecision.value_counts())
	print(y1)
	subDf = df[df.techCat1 == 'Storage']
	y2 = (subDf.FinalDecision.value_counts())
	print(df.FinalDecision)

	plt.figure(figsize=(4,4.75))
	plt.subplot(position = [0.15, 0.1, 0.85, 0.85])
	plt.bar(range(3), y1)
	plt.bar(range(3), y2, color = c1[1])
	plt.ylabel('Frequency')
	plt.xticks(range(3), ('Pivot', 'Perish', 'Persist'))
	plt.xlim(-0.5, 3.5)
	plt.text(2.45, 5, 'Storage')
	plt.text(2.45, 57, 'Other\nCategories')
	plt.savefig('outcomeStorage.png', dpi = 300)
#makeFrequencyBarChart(df)
