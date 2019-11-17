import numpy as np 
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize 
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex
from matplotlib.colors import to_rgb
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
print(df.columns)
#df.awardAmount = np.log(df.awardAmount)


# calculate the chi2 based on open/designed outcomes 
stat, p, dof, expected = chi2calc(df, 'FinalDecision')
print('stat', stat, 'p', p, 'dof', dof, 'expected', expected)

def makeAMETables(modmg): #makes a csv version of the odds ratio 
	outdf = pd.DataFrame(columns = ('Parameter', 'Perish', 'Persist', 'Pivot'))
	outdf[0]['Parameter'] = modmg(['dy/dx'][('FinalDecision=Perish', 'OPEN')])
	print(outdf)


	print('hello world')

# define some regression models 

def runMod1(pooling): #just open
	if pooling == 'total':
		exog = (df[['OPEN']]) #sm.add_constant
	elif pooling == 'none':
		exog = (df[['OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17']])
	elif pooling == 'fakePartial':
		exog = (df[['OPEN', 'early', 'middle', 'late']])
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
		exog = df[['ForProf', 'OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17']]
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
		exog = df[['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17']]
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
		exog = df[['Partners', 'OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17']]
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
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'Partners', 'OPEN', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17']]
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
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'Partners', 'dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17']]
	elif pooling == 'fakePartial':
		exog = df[['awardAmount', 'ForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'Partners', 'early', 'middle', 'late']]
	else:
		print('hello world')

	mdl7 = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit()
	print(mdl7.summary())
	mod7mg = mdl7.get_margeff()
	print(mod7mg.summary())
	return(mod7mg.summary_frame())

#runMod1('total')
#runMod2('total')
runMod3('total')
#y = runMod4('total')
#runMod5('total')
#y2 = runMod6('total')
#runMod7('total')

#x2 = runMod2('none')
#x3 = runMod2('fakePartial')
cmap = plt.get_cmap("tab10")
print(cmap)

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

def makeBarChart(df):
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
#makeBarChart(df)
