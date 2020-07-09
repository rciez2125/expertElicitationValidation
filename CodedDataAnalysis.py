import numpy as np 
import pandas as pd 
pd.set_option('display.max_columns', 10)
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
import codedDataAnalysisScripts
from scrips import chi2calc


# load & clean the final compiled dataset
df = pd.read_csv('Data/FinalData_cleaned.csv')
d = codedDataAnalysisScripts.codedDataAnalysisScripts('a')
df = d.cleanData(df)
print(df.columns)
print(df.companies.unique())

duplicateRowsDF = df[df.duplicated(['companies'])]
print(duplicateRowsDF)
duplicateRowsDF.to_csv('Duplicates.csv')

#d.makeSummaryFigures(df, 'ppt')

## 1) univariable analysis of each independent variable 
def runUnivariableAnalysis(df):
	d.univariateCategorical_2Categories(df)
	# nonstandard number of categories
	d.univariateCategorical_multipleCategories(df) # this works with update, but look at startups
	# continous variable test 
	d.univariateContinuous(df) # this works with update

	# based on this analysis, which variables matter?
	# OPEN 						No
	# Partners 					No
	# Tech Category 			Yes
	# Recipinet Type (binary)	Yes
	# Recipient type (3)		Yes
	# Start year (every year)	Yes
	# Start year (grouped)		Yes
	# Award Amount 				Yes 
#runUnivariableAnalysis(df)

### 2) Fit a multivariate model with variables identified in step 1
# exclude open, partners
# optionsf on recipient type and start year

#print(df.groupby(['SmallForProf', 'techCat1', 'startYr']).size())
#print(df.groupby(['LargeForProf', 'techCat1', 'startYr']).size())

def makeModelStep2(time, forProfType, timeBaseline):
	ex_list = ['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT']
	if forProfType == 'bySize':
		if time == 'every year':
			#ex_list = ['SmallForProf', 'LargeForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT']
			#ex_list.extend(())
			if timeBaseline == 'late':
				ex_list = ['dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'SmallForProf', 'LargeForProf', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'awardAmount']# 'awardAmount'))
			elif timeBaseline == 'early':
				ex_list.extend(('dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17'))#, 'awardAmount'))
		elif time == 'grouped':
			ex_list.extend(('SmallForProf', 'LargeForProf'))
			if timeBaseline == 'late':
				ex_list.extend(('awardAmount', 'early', 'middle'))
			elif timeBaseline == 'early':
				ex_list.extend(('awardAmount', 'middle', 'late'))
		else:
			ex_list.extend(('SmallForProf', 'LargeForProf'))
			ex_list.append('awardAmount')
	elif forProfType == 'byStartup':
		ex_list = ['awardAmount', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT']
		ex_list.extend(('StartupForProf', 'OtherForProf'))
		if time == 'every year': 
			if timeBaseline == 'late':
				ex_list.extend(('dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16'))
			elif timeBaseline == 'early': 
				ex_list.extend(('dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17'))
		elif time == 'grouped':
			if timeBaseline == 'late':
				ex_list.extend(('early', 'middle'))
			elif timeBaseline == 'early':
				ex_list.extend(('middle', 'late'))
	else: 
		ex_list.extend(('ForProf', 'awardAmount'))
		if time == 'every year': 
			if timeBaseline == 'late':
				ex_list.extend(('dum09', 'dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16'))
			elif timeBaseline == 'early':
				ex_list.extend(('dum10', 'dum11', 'dum12', 'dum13', 'dum14', 'dum15', 'dum16', 'dum17'))
		elif time == 'grouped':
			if timeBaseline == 'late':
				ex_list.extend(('early', 'middle'))
			elif timeBaseline == 'early':
				ex_list.extend(('middle', 'late'))
	exog = df[ex_list]	
	mod = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)
	#print(mod.summary())
	modmg = mod.get_margeff(at = 'overall')
	return(mod)

def checkForProfitModels(timeBaseline):
	print('Every year, different for-prof')
	#m1 = makeModelStep2('every year', 'bySize', timeBaseline) # works if award amount is first 
	#m2 = makeModelStep2('every year', 'byStartup', timeBaseline) # works if award amount is first # works
	#m3 = makeModelStep2('every year', 'two', timeBaseline) # works if award amount is before years #works
	#print(m1.llf)
	#print(m2.llf)
	#d.modelSelectionComparison(m1, m3, 'ModelSelectionData/EveryYear_bySizeForProf.csv')
	#d.modelSelectionComparison(m2, m3, 'ModelSelectionData/EveryYear_byStartupForProf.csv')

	print('Grouped years, different for-prof')
	m4 = makeModelStep2('grouped', 'bySize', timeBaseline) # works regardless of order # works
	m5 = makeModelStep2('grouped', 'byStartup', timeBaseline) # works regardless of order # works
	m6 = makeModelStep2('grouped', 'two', timeBaseline) # this had the highest percentage  #works
	print(m4.summary())
	print(m4.llf)
	print(m5.llf)
	d.modelSelectionComparison(m4, m6, 'ModelSelectionData/GroupedYear_bySizeForProf.csv')
	d.modelSelectionComparison(m5, m6, 'ModelSelectionData/GroupedYear_byStartupForProf.csv')

	print('No time data, different for-prof')
	m7 = makeModelStep2('none', 'bySize', timeBaseline) # works
	m8 =makeModelStep2('none', 'byStartup', timeBaseline) # works
	m9 = makeModelStep2('none', 'two', timeBaseline) #works
	print(m7.llf)
	print(m8.llf)
	d.modelSelectionComparison(m7, m9, 'ModelSelectionData/NoYear_bySizeForProf.csv')
	d.modelSelectionComparison(m8, m9, 'ModelSelectionData/NoYear_byStartupForProf.csv')
#checkForProfitModels('late')

def makeTimePlot(mod, figName, baseTime):
	if baseTime == 'earlyAll':
		t = np.linspace(2009, 2016, 8)
		betas = np.vstack((np.asarray(mod.params.loc['dum10']), np.asarray(mod.params.loc['dum11']), np.asarray(mod.params.loc['dum12']), np.asarray(mod.params.loc['dum13']), np.asarray(mod.params.loc['dum14']), np.asarray(mod.params.loc['dum15']), np.asarray(mod.params.loc['dum16']), np.asarray(mod.params.loc['dum17'])))
		ses = np.vstack((np.asarray(mod.bse.loc['dum10']), np.asarray(mod.bse.loc['dum11']), np.asarray(mod.bse.loc['dum12']), np.asarray(mod.bse.loc['dum13']), np.asarray(mod.bse.loc['dum14']), np.asarray(mod.bse.loc['dum15']), np.asarray(mod.bse.loc['dum16']), np.asarray(mod.bse.loc['dum17'])))
		st = 'Late Baseline'
	elif baseTime == 'lateAll': 
		t = np.linspace(2010, 2017, 8)
		betas = np.vstack((np.asarray(mod.params.loc['dum09']), np.asarray(mod.params.loc['dum10']), np.asarray(mod.params.loc['dum11']), np.asarray(mod.params.loc['dum12']), np.asarray(mod.params.loc['dum13']), np.asarray(mod.params.loc['dum14']), np.asarray(mod.params.loc['dum15']), np.asarray(mod.params.loc['dum16']), ))
		ses = np.vstack((np.asarray(mod.bse.loc['dum09']), np.asarray(mod.bse.loc['dum10']), np.asarray(mod.bse.loc['dum11']), np.asarray(mod.bse.loc['dum12']), np.asarray(mod.bse.loc['dum13']), np.asarray(mod.bse.loc['dum14']), np.asarray(mod.bse.loc['dum15']), np.asarray(mod.bse.loc['dum16'])))
		st = 'Early Baseline'
	elif baseTime == 'earlyGrouped':
		t = (2013,2014)
		betas = np.vstack((np.asarray(mod.params.loc['middle']), np.asarray(mod.params.loc['late'])))
		ses = np.vstack((np.asarray(mod.bse.loc['middle']), np.asarray(mod.bse.loc['late'])))
		st = 'Early Baseline'
	elif baseTime == 'lateGrouped':	
		t = (2010, 2013)
		betas = np.vstack((np.asarray(mod.params.loc['early']), np.asarray(mod.params.loc['middle'])))
		ses = np.vstack((np.asarray(mod.bse.loc['early']), np.asarray(mod.bse.loc['middle'])))
		st = 'Late Baseline'
	else:
		print('not a valid time scheme')


	plt.figure(figsize=(5,3))
	ax1 = plt.subplot(position = [0.17, 0.15, 0.35, 0.7])
	plt.plot(t, betas[:,0], '.b')
	for n in range(len(t)):
		plt.plot([t[n], t[n]], [betas[n,0] - 1.96*ses[n,0], betas[n,0] + 1.96*ses[n,0]], '-b')
	plt.plot([2008, 2019], [0,0], '-k')
	plt.ylim(-5,5)
	plt.title('Persist')
	plt.ylabel('Coefficients with 95%\nConfidence Intervals')
	ax2 = plt.subplot(position = [0.6, 0.15, 0.35, 0.7])
	plt.plot(t, betas[:,1], '.b')
	for n in range(len(t)):
		plt.plot([t[n], t[n]], [betas[n,1] - 1.96*ses[n,1], betas[n,1] + 1.96*ses[n,1]], '-b')
	plt.plot([2008, 2019], [0,0], '-k')
	plt.ylim(-5,5)
	plt.title('Pivot')
	plt.suptitle(st)
	plt.savefig(figName, dpi=300)

def checkTimeBaseline():
	m1L = makeModelStep2('every year', 'byStartup', 'late') # works if award amount is first 
	#print(m1L.summary())
	m3L = makeModelStep2('every year', 'two', 'late') # works if award amount is before years
	m5L = makeModelStep2('grouped', 'byStartup', 'late') # works regardless of order
	m6L = makeModelStep2('grouped', 'two', 'late') # this had the highest percentage

	print('starting early')
	m1E = makeModelStep2('every year', 'byStartup', 'early') # works if award amount is first 
	#print(m1E.summary())
	m3E = makeModelStep2('every year', 'two', 'early') # works if award amount is before years
	m5E = makeModelStep2('grouped', 'byStartup', 'early') # works regardless of order
	m6E = makeModelStep2('grouped', 'two', 'early') # this had the highest percentage
	
	makeTimePlot(m1L, 'ModelSelectionData/everyYear_Startup_late.png', 'lateAll')
	makeTimePlot(m1E, 'ModelSelectionData/everyYear_Startup_early.png', 'earlyAll')
	print(m1L.llf, m1E.llf)

	makeTimePlot(m5L, 'ModelSelectionData/grouped_Startup_late.png', 'lateGrouped')
	makeTimePlot(m5E, 'ModelSelectionData/grouped_Startup_early.png', 'earlyGrouped')

	makeTimePlot(m3L, 'ModelSelectionData/everyYear_forprof_late.png', 'lateAll')
	makeTimePlot(m3E, 'ModelSelectionData/everyYear_forprof_early.png', 'earlyAll')	

	makeTimePlot(m6L, 'ModelSelectionData/grouped_forprof_late.png', 'lateGrouped')
	makeTimePlot(m6E, 'ModelSelectionData/grouped_forprof_early.png', 'earlyGrouped')
	print(m6L.llf, m6E.llf)
#checkTimeBaseline()
# using the earliest timeframe as the baseline is better than using the most recent projects as baseline 

#makeTimePlot(mg, 'testyrs_earliest.png', 'earlyAll')
def checkTimes():
	print('Does time matter?')
	m2 = makeModelStep2('every year', 'byStartup', 'early') # works if award amount is first 
	m3 = makeModelStep2('every year', 'two', 'early') # works if award amount is before years
	m5 = makeModelStep2('grouped', 'byStartup', 'early') # works regardless of order
	m6 = makeModelStep2('grouped', 'two', 'early') # this had the highest percentage
	m8 = makeModelStep2('none', 'byStartup', 'na')
	m9 = makeModelStep2('none', 'two', 'na')

	print(m2.llf)
	print(m5.llf)
	print(m8.llf)
	d.modelSelectionComparison(m2, m5, 'ModelSelectionData/byStartup_EveryYearGroupedYear.csv')
	d.modelSelectionComparison(m2, m8, 'ModelSelectionData/byStartup_EveryYearNoYear.csv')
	d.modelSelectionComparison(m5, m8, 'ModelSelectionData/byStartup_GroupedYearNoYear.csv')
	# more granular time is always better, jump is biggest from none-->every year, then grouped --> every year, then none-->grouped

	print(m3.llf)
	print(m6.llf)
	print(m9.llf)
	d.modelSelectionComparison(m3, m6, 'ModelSelectionData/forprof_EveryYearGroupedYear.csv')
	d.modelSelectionComparison(m3, m9, 'ModelSelectionData/forprof_EveryYearNoYear.csv')
	d.modelSelectionComparison(m6, m9, 'ModelSelectionData/forprof_GroupedYearNoYear.csv')
	# more granular time is always better, jump is biggest from none-->every year, then grouped --> every year, then none-->grouped
#checkTimes()

# what's happening in 2011? --> there are no projects in 2011 that persisted --> have to do some kind of grouping 
# bins track with OPEN calls

# does the baseline technology category matter? we started with stationary storage? 
def checkTechCatBaseline():
	print(df.groupby(['CompanyType']).size())
	print(51/(df.shape[0]))
	print(51/(df.shape[0] - 246))
	print(df.groupby(['CompanyType', 'FinalDecision']).size())

	s = (df.groupby(['techCat1']).size()/(0.01*df.shape[0]))
	print(s)
	print(s.loc['Storage'])
	ex_list = ['awardAmount', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_SS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'StartupForProf', 'OtherForProf', 'middle', 'late']

	def dropRun(vartoDrop, m):
		m = ex_list.copy()
		m.remove(vartoDrop)
		exog = df[m]	
		mod = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)
		return(mod)
	t1 = dropRun('TC_SS', ex_list)
	t2 = dropRun('TC_DG', ex_list)
	t3 = dropRun('TC_RE', ex_list)
	t4 = dropRun('TC_TF', ex_list)
	t5 = dropRun('TC_TS', ex_list)
	print(t1.llf, t2.llf, t3.llf, t4.llf, t5.llf)


	dataLabs = ['Storage', 'DG', 'RE', 'TF', 'TS']
	xplot = [s.loc['Storage'], s.loc['Distributed Generation'], s.loc['Resource Efficiency'], s.loc['Transportation Fuels'], s.loc['Transportation Storage']]
	yplot = [t1.llf, t2.llf, t3.llf, t4.llf, t5.llf]
	plt.figure(figsize=(5,3))
	plt.plot(xplot, yplot, '.')
	plt.xlabel('share of all projects')
	plt.ylabel('log likelihood')
	for n in range(5):
		plt.text(xplot[n], yplot[n]-0.5, dataLabs[n], horizontalalignment = 'center', fontsize = 6)
	plt.ylim(-465, -455)
	plt.savefig('ModelSelectionData/techCatplots.png', dpi=300)

	# slight differences, but nothing major 
#checkTechCatBaseline()

def addOneVariable(time, variableToAdd):
	ex_list = ['awardAmount', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'StartupForProf', 'OtherForProf']
	if time == 'grouped':
		ex_list.extend(('middle', 'late'))
	ex_list.extend(variableToAdd)
	print(ex_list)
	exog = df[ex_list]	
	mod = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)
	print(mod.summary())
	modmg = mod.get_margeff(at = 'overall')
	return(mod)
def tryOtherVars():
	m1 = addOneVariable('grouped', ['OPEN'])
	m2 = addOneVariable('none', ['OPEN'])
	m3 = addOneVariable('grouped', ['Partners'])
	m4 = addOneVariable('none', ['Partners'])

	m5 = makeModelStep2('grouped', 'byStartup', 'early')
	m6 = makeModelStep2('none', 'byStartup', 'na')
	d.modelSelectionComparison(m1, m5, 'ModelSelectionData/grouped_Startup_withOpen.csv') # adding open doesn't change any of the other betas by more than 20% 
	d.modelSelectionComparison(m3, m5, 'ModelSelectionData/grouped_Startup_withPartners.csv') # adding open doesn't change any of the other betas by more than 20% 

	#partners is never significant
	# open is never significant --> leave it in for our analysis

	print(m1.summary())
	print(m3.summary())
#tryOtherVars()

# step 4
# there are a few parameters that change when we add partners and open, they aren't significant parameters
def buildPreliminaryMEmodel(time):
	ex_list = ['awardAmount', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'StartupForProf', 'OtherForProf']
	if time == 'grouped':
		ex_list.extend(('middle', 'late'))
	ex_list.append('OPEN')
	exog = df[ex_list]	
	mod = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)
	#print(mod.summary())
	modmg = mod.get_margeff(at = 'overall')
	#print(modmg.summary())
	return(mod)

def contModelForm():
	m1 = buildPreliminaryMEmodel('grouped')
	# step 5
	# for each continuous variable, we have to check the assumption that the logit increases/decreases linearly as a function of the covariate
	# start with a scatterplot
	# obtain the quartiles of the distribution of the continuous variable
	#print(df.awardAmount.describe())
	x = df.awardAmount.describe()
	# create categorical variables with four levels using 3 cutpoints based on cutpoints
	df['awardAmtQ1']=0
	df['awardAmtQ2']=0
	df['awardAmtQ3']=0
	df['awardAmtQ4']=0

	print(df.shape[0])
	for n in range(df.shape[0]):
		if df.awardAmount[n]<= x['25%']:
			df.awardAmtQ1[n] = 1
		elif df.awardAmount[n]<= x['50%']:
			df.awardAmtQ2[n] = 1
		elif df.awardAmount[n]<= x['75%']:
			df.awardAmtQ3[n] = 1
		else: 
			df.awardAmtQ4[n] = 1

	def awardAmtQuartile(time):
		ex_list = ['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'ForProf', 'awardAmtQ2', 'awardAmtQ3', 'awardAmtQ4']
		if time == 'grouped':
			ex_list.extend(('middle', 'late'))
		ex_list.append('OPEN')
		exog = df[ex_list]	
		mod = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)
		modmg = mod.get_margeff(at = 'overall')
		return(mod)
	
	# fit multivariable model replacing continuous variable with quartile groups (lowest = reference)
	m2 = awardAmtQuartile('grouped')

	def compareAwardModels(cM, contVar, lM, quartData, figName, log):
		x_c = np.asarray([(quartData['50%']-quartData['25%'])/2 + quartData['25%'], (quartData['75%']-quartData['50%'])/2 + quartData['50%'], (quartData['max']-quartData['75%'])/2 + quartData['75%']])
		x_cl = np.asarray([(quartData['25%']-quartData['min'])/2 + quartData['min'], (quartData['50%']-quartData['25%'])/2 + quartData['25%'], (quartData['75%']-quartData['50%'])/2 + quartData['50%'], (quartData['max']-quartData['75%'])/2 + quartData['75%']])
		persistQuartData = np.asarray([lM.params.loc['awardAmtQ2'][0], lM.params.loc['awardAmtQ3'][0], lM.params.loc['awardAmtQ4'][0]])
		persistQuartLB = persistQuartData - 1.96*np.asarray([lM.bse.loc['awardAmtQ2'][0], lM.bse.loc['awardAmtQ3'][0], lM.bse.loc['awardAmtQ4'][0]])
		persistQuartUB = persistQuartData + 1.96*np.asarray([lM.bse.loc['awardAmtQ2'][0], lM.bse.loc['awardAmtQ3'][0], lM.bse.loc['awardAmtQ4'][0]])
		pivotQuartData = np.asarray([lM.params.loc['awardAmtQ2'][1], lM.params.loc['awardAmtQ3'][1], lM.params.loc['awardAmtQ4'][1]])
		pivotQuartLB = pivotQuartData - 1.96*np.asarray([lM.bse.loc['awardAmtQ2'][1], lM.bse.loc['awardAmtQ3'][1], lM.bse.loc['awardAmtQ4'][1]])
		pivotQuartUB = pivotQuartData + 1.96*np.asarray([lM.bse.loc['awardAmtQ2'][1], lM.bse.loc['awardAmtQ3'][1], lM.bse.loc['awardAmtQ4'][1]])

		if log == 0:
			beta_hat_persist = x_cl*cM.params.loc[contVar][0]
			beta_hat_persist[0] = 0
			beta_hat_pivot = x_cl*cM.params.loc[contVar][1]
			beta_hat_pivot[0] = 0
			st = 'Linear Model'
			xl = 'Linear'

		else: 
			beta_hat_persist = np.log10(x_c)*cM.params.loc[contVar][0]
			beta_hat_pivot = np.log(x_c)*cM.params.loc[contVar][1]
			x_cl = x_c
			st = 'Log Model'
			xl = 'Log'
		# plot estimated coefficients for quartile 
		plt.figure(figsize=(5,3))
		ax1 = plt.subplot(position = [0.15, 0.15, 0.27, 0.7])
		plt.plot(x_c, persistQuartData, 'x-b')
		plt.plot(x_cl, beta_hat_persist, 'o-g')
		for n in range(len(x_c)):
			plt.plot([x_c[n], x_c[n]], [persistQuartLB[n], persistQuartUB[n]], '-b', linewidth = 0.5)
			plt.plot([x_c[n]-0.1, x_c[n]+0.1], [persistQuartLB[n], persistQuartLB[n]], '-b', linewidth = 0.5)
			plt.plot([x_c[n]-0.1, x_c[n]+0.1], [persistQuartUB[n], persistQuartUB[n]], '-b', linewidth = 0.5)
		#plt.plot(x_c, persistQuartLB, 'xb')
		#plt.plot(x_c, persistQuartUB, 'xb')
		
		plt.ylim(-1, 2)
		plt.xlabel('Award Amount')
		plt.xlim(-1,7)
		plt.ylabel('Predicted')
		plt.title('Persist')

		ax2 = plt.subplot(position = [0.53, 0.15, 0.27, 0.7])
		plt.plot(x_c, pivotQuartData, 'x-b')
		plt.plot(x_cl, beta_hat_pivot, 'o-g')
		for n in range(len(x_c)):
			plt.plot([x_c[n], x_c[n]], [pivotQuartLB[n], pivotQuartUB[n]], '-b', linewidth = 0.5)
			plt.plot([x_c[n]-0.1, x_c[n]+0.1], [pivotQuartLB[n], pivotQuartLB[n]], '-b', linewidth = 0.5)
			plt.plot([x_c[n]-0.1, x_c[n]+0.1], [pivotQuartUB[n], pivotQuartUB[n]], '-b', linewidth = 0.5)
		#plt.plot(x_c, pivotQuartLB, 'xb')
		#plt.plot(x_c, pivotQuartUB, 'xb')
		
		plt.legend(('Quartiles', xl), fontsize = 6, loc = 'center left', bbox_to_anchor=(1.04, 0.5),)
		plt.ylim(-1, 2)
		plt.xlabel('Award Amount')
		plt.xlim(-1,7)
		plt.title('Pivot')
		plt.suptitle(st)

		plt.savefig(figName, dpi=300)

	compareAwardModels(m1, 'awardAmount', m2, x, 'ModelSelectionData/AwardAmountTestLinearFit.png', 0)

	# try a log
	df['logAward'] = np.log10(df.awardAmount)
	def awardAmtLog(time):
		ex_list = ['logAward', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'StartupForProf', 'OtherForProf']
		if time == 'grouped':
			ex_list.extend(('middle', 'late'))
		ex_list.append('OPEN')
		exog = df[ex_list]
		mod = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)
		#print(mod.summary())
		modmg = mod.get_margeff(at = 'overall')
		return(mod)
	m3 = awardAmtLog('grouped')
	print(m1.llf, m2.llf, m3.llf)

	compareAwardModels(m3, 'logAward', m2, x, 'ModelSelectionData/AwardAmountTestLogFit.png', 1)
	# the log transformation is less significant/not much better than linear --> skip it 
#contModelForm()

# check for interactions 
def interactionCheck(df):
	a = df.groupby(['techCat1', 'StartupForProf']).size()
	#print(a) # --> lots of holes where there isn't an interaction term
	b = df.groupby(['techCat1', 'startYr']).size()
	#print(b)
	c = df.groupby(['techCat1', 'OPEN']).size() # --> lots of holes

	def awardAmtOPEN(df):
		plt.figure()
		plt.plot(df['awardAmount'], df['OPEN'], '.')
		plt.savefig('ModelSelectionData/awardAmtOpen.png', dpi = 300)
		plt.clf()
		df['OPENAward'] = df['OPEN']*df['awardAmount']
		m = addOneVariable('grouped', ('OPEN', 'OPENAward')) # not significant 
		return df
	#df = awardAmtOPEN(df)

	def awardAmtCoType(df):
		plt.figure()
		plt.subplot(1,3,1)
		x = df[df['StartupForProf']==1]
		y = df[df['OtherForProf']==1]
		z = df[df['StartupForProf']==0]
		z = z[z['OtherForProf']==0]

		plt.hist(x['awardAmount'])
		plt.subplot(1,3,2)
		plt.hist(y['awardAmount'])
		#plt.plot(df['awardAmount'], df['OtherForProf'], '.')
		plt.subplot(1,3,3)
		plt.hist(z['awardAmount'])
		plt.savefig('ModelSelectionData/awardAmtStartup.png', dpi = 300)
		plt.clf()
		return df 
	#df = awardAmtCoType(df)

	def CoTypeOPEN(df):
		df['OPENStartup'] = df.OPEN*df.StartupForProf
		df['OPENotherForProf'] = df.OPEN*df.OtherForProf
		m1 = addOneVariable('grouped', ('OPEN', 'OPENStartup', 'OPENotherForProf')) # only significant interaction is on OPEN + other for-profit and persist (negative) --> do any other betas change? 
		m2 = addOneVariable('grouped', ['OPEN'])
		d.modelSelectionComparison(m1, m2, 'ModelSelectionData/interactionsOPENCoType.csv')
		return df
	df = CoTypeOPEN(df)

	def CoTypeTime(df):
		# company type + time 
		s = df[df['StartupForProf']==1]
		sg = s.groupby(['FinalDecision', 'early', 'middle', 'late']).size()
		print(sg)

		o = df[df['OtherForProf'] == 1]
		og = o.groupby(['FinalDecision', 'early', 'middle', 'late']).size()
		print(og)

		n = df[df['StartupForProf']==0]
		n = n[n['OtherForProf']==0]
		ng = n.groupby(['FinalDecision', 'early', 'middle', 'late']).size()
		print(ng)

		df['StartupMiddle'] = df.StartupForProf*df.middle
		df['StartupLate'] = df.StartupForProf*df.late
		df['otherForprofMiddle'] = df.OtherForProf*df.middle
		df['otherforprofLate'] = df.OtherForProf*df.late
		m1 = addOneVariable('grouped', ('OPEN', 'StartupMiddle', 'StartupLate', 'otherForprofMiddle', 'otherforprofLate')) # startup middle only one that was significant --> do any other variables change?
		m2 = addOneVariable('grouped', ['OPEN'])
		d.modelSelectionComparison(m1, m2, 'ModelSelectionData/interactionsCoTypeTime.csv')
		return df
	df = CoTypeTime(df)

	# time and award amount
	def awardAmtTime(df):
		fig, ax1 = plt.subplots()
		data = [np.asarray(df[df['startYr']==0].awardAmount), np.asarray(df[df['startYr']==1].awardAmount), np.asarray(df[df['startYr']==2].awardAmount), np.asarray(df[df['startYr']==3].awardAmount), np.asarray(df[df['startYr']==4].awardAmount), np.asarray(df[df['startYr']==5].awardAmount), np.asarray(df[df['startYr']==6].awardAmount), np.asarray(df[df['startYr']==7].awardAmount), np.asarray(df[df['startYr']==8].awardAmount)] 
		totalAwards = df.groupby('startYr')['awardAmount'].sum()
		totalAwards.index +=1

		ax1.boxplot(data)
		ax1.set_ylabel('Individual Award Amounts')
		ax1.set_xticks([1,2,3,4,5,6,7,8,9], ('2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'))
		
		ax2 = ax1.twinx()
		ax2.plot(totalAwards)
		#ax2.set_xticks([1,2,3,4,5,6,7,8,9], 
		ax2.set_xticklabels(('2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'))
		#plt.plot(df.startYr, df.awardAmount, '.')
		ax2.set_ylabel('Total Annual Awards')
		plt.savefig('ModelSelectionData/YearAwardAmt.png', dpi = 300)
		plt.clf()
		# 

		df['AwardMiddle'] = df.awardAmount * df.middle
		df['AwardLate'] = df.awardAmount * df.late
		m = addOneVariable('grouped', ('OPEN', 'AwardMiddle', 'AwardLate')) # neither of these are significant
	#df = awardAmtTim(df)

	def openTime(df):
		df['OPENmiddle'] = df.OPEN * df.middle
		df['OPENlate'] = df.OPEN * df.late
		m = addOneVariable('grouped', ('OPEN', 'OPENmiddle', 'OPENlate')) # several of these are significant (for persisting and perishing) --> what does this mean?
		return(df)
	#df = openTime(df)

	def techCatAwardAmt(df):
		fig, ax1 = plt.subplots(figsize=(6,6))# make boxplots of award amounts by tech category 
		ax1.set_position([0.2, 0.2, 0.7, 0.7])
		data = [np.asarray(df[df['TC_SS']==1].awardAmount), np.asarray(df[df['TC_TF']==1].awardAmount), np.asarray(df[df['TC_DG']==1].awardAmount), np.asarray(df[df['TC_TS']==1].awardAmount), np.asarray(df[df['TC_BE']==1].awardAmount), np.asarray(df[df['TC_RE']==1].awardAmount), np.asarray(df[df['TC_ME']==1].awardAmount), np.asarray(df[df['TC_EE']==1].awardAmount), np.asarray(df[df['TC_GR']==1].awardAmount), np.asarray(df[df['TC_OT']==1].awardAmount)]
		totalAwards = df.groupby('techCat1')['awardAmount'].sum()
		totalAwards = totalAwards.to_frame()
		t = sum((totalAwards.loc['Transportation Network'], totalAwards.loc['Transportation Vehicles'], totalAwards.loc['Centralized Generation']))
		totalAwards.loc['Other'] = t
		totalAwards = totalAwards.drop(['Transportation Network', 'Transportation Vehicles', 'Centralized Generation'])
		totalAwards = totalAwards.reindex(['Storage', 'Transportation Fuels', 'Distributed Generation', 'Transportation Storage', 'Building Efficiency', 'Resource Efficiency', 'Manufacturing Efficiency', 'Electrical Efficiency', 'Grid', 'Other'])

		ax1.boxplot(data)
		ax1.set_ylabel('Individual Award Amounts')
		ax1.set_xticklabels(('Storage', 'Transport Fuels', 'Dist Gen', 'Transport Storage', 'Building Eff', 'Resource Eff', 'Mfg Eff', 'Elec Eff', 'Grid', 'Other'), rotation = 'vertical')
		ax2 = ax1.twinx()
		ax2.set_position([0.2, 0.2, 0.7, 0.7])
		ax2.plot(np.arange(10)+1, totalAwards.awardAmount)
		
		ax2.set_ylabel('Total Program Awards')
		plt.savefig('ModelSelectionData/TechCatAwardAmt.png', dpi = 300)
		plt.clf()

		df['TC_TFAward'] = df.TC_TF*df.awardAmount
		df['TC_DGAward'] = df.TC_DG*df.awardAmount
		df['TC_TSAward'] = df.TC_TS*df.awardAmount
		df['TC_BEAward'] = df.TC_BE*df.awardAmount
		df['TC_REAward'] = df.TC_RE*df.awardAmount
		df['TC_MEAward'] = df.TC_ME*df.awardAmount
		df['TC_EEAward'] = df.TC_EE*df.awardAmount
		df['TC_GRAward'] = df.TC_GR*df.awardAmount
		df['TC_OTAward'] = df.TC_OT*df.awardAmount

		m1 = addOneVariable('grouped', ('OPEN', 'TC_TFAward', 'TC_DGAward', 'TC_TSAward', 'TC_BEAward', 'TC_REAward', 'TC_MEAward', 'TC_EEAward', 'TC_GRAward', 'TC_OTAward'))
		# only the transportation fuels and distributed generation projects had a significant negative correlation with pivoting (ie more money was bad)
		m2 = addOneVariable('grouped', ['OPEN'])
		d.modelSelectionComparison(m1, m2, 'ModelSelectionData/interactionsTechCatAwardAmt.csv')
		return(df)
	df = techCatAwardAmt(df)	
#interactionCheck(df)

def displayModelResults():
	buildPreliminaryMEmodel('grouped')
	m = buildPreliminaryMEmodel('grouped')
	me = m.get_margeff(at = 'overall')
	print(me.margeff)
	print(me.margeff_se)
	# perish, persist, pivot
	print(me.summary())

	
	def makeBarFig(format, varsToDisplay, figName):
		barLabels = ('Award Amount', 'Trans Fuels', 'Dist Gen', 'Trans Stor', 'Bldg Eff', 'Resource Eff', 'Mfg Eff', 'Elec Eff', 'Grid', 'Other', 'Startup', 'Other For-Prof', 'Cohort 2', 'Cohort 3', 'OPEN')
		titles = ('Perish', 'Persist', 'Pivot')
		spots = [[0.1, 0.02, 0.25, 0.89], [0.4, 0.02, 0.25, 0.89], [0.7, 0.02, 0.25, 0.89]]

		if format == 'ppt':
			fgs = (11.5, 5) # presentation
			fs = 14
		else:
			fgs = (7, 4)
			fs = 7
		plt.figure(figsize = fgs)

		
		for m in range(3):
			plt.subplot(position = spots[m])
			if varsToDisplay == 'OPEN':
				rect = plt.Rectangle([-0.99, -0.7], 14.5, 1.25, facecolor = [1,1,1], alpha = 0.8, zorder = 3)
				rect2 = plt.Rectangle([50, -0.7], 14.5, 1.25, facecolor = [1,1,1], alpha = 0.8, zorder = 3)
			elif varsToDisplay == 'cohort':
				rect = plt.Rectangle([-0.99, -0.7], 12.5, 1.25, facecolor = [1,1,1], alpha = 0.8, zorder = 3)
				rect2 = plt.Rectangle([13.5, -0.7], 1, 1.25, facecolor = [1,1,1], alpha = 0.8, zorder = 3)
			elif varsToDisplay == 'companyType':
				rect = plt.Rectangle([-0.99, -0.7], 10.5, 1.25, facecolor = [1,1,1], alpha = 0.8, zorder = 3)
				rect2 = plt.Rectangle([11.5, -0.7], 3, 1.25, facecolor = [1,1,1], alpha = 0.8, zorder = 3)

			elif varsToDisplay == 'techCat':
				rect = plt.Rectangle([-0.99, -0.7], 1.5, 1.25, facecolor = [1,1,1], alpha = 0.85, zorder = 3)
				rect2 = plt.Rectangle([9.5, -0.7], 5, 1.25, facecolor = [1,1,1], alpha = 0.85, zorder = 3)
			elif varsToDisplay == 'awardAmount':
				rect = plt.Rectangle([-3, -0.7], 1, 1.25, facecolor = [1,1,1], alpha = 0.8, zorder = 3)
				rect2 = plt.Rectangle([0.5, -0.7], 14, 1.25, facecolor = [1,1,1], alpha = 0.8, zorder = 3)
			else:
				rect = plt.Rectangle([-3, -0.7], 1, 1.25, facecolor = [1,1,1], alpha = 0.8, zorder = 3)
				rect2 = plt.Rectangle([-30, -0.7], 1, 1.25, facecolor = [1,1,1], alpha = 0.8, zorder = 3)

			rect3 = plt.Rectangle([0.5, -0.7], 9, 1.25, facecolor = [1,1,1], alpha = 0.3, zorder = 3)
			rect4 = plt.Rectangle([3.5, -0.7], 6, 1.25, facecolor = [1,1,1], alpha = 0.3, zorder = 3)
			rect5 = plt.Rectangle([0.5, -0.7], 2, 1.25, facecolor = [1,1,1], alpha = 0.3, zorder = 3)
			rect6 = plt.Rectangle([3.5, -0.7], 6, 1.25, facecolor = [1,1,1], alpha = 0.3, zorder = 3)

			plt.bar(np.arange(15), me.margeff[:,m])
			for n in range(15):
				plt.plot([n, n], [me.margeff[n,m]-1.96*me.margeff_se[n,m], me.margeff[n,m]+1.96*me.margeff_se[n,m]], '-k', zorder = 1)
				if me.margeff[n, m]>0:
					plt.text(n, me.margeff[n,m]+1.96*me.margeff_se[n,m]+0.015, barLabels[n], rotation = 90, horizontalalignment = 'center', verticalalignment= 'bottom', fontsize = fs - 4, zorder = 2)
				else:
					plt.text(n, me.margeff[n,m]-1.96*me.margeff_se[n,m]-0.015, barLabels[n], rotation = 90, horizontalalignment = 'center', verticalalignment= 'top', fontsize = fs - 4, zorder = 2)
			plt.title(titles[m], fontsize = fs)
			plt.ylim(-0.71, 0.56)
			plt.xticks([], [])

			plt.gca().add_patch(rect)
			plt.gca().add_patch(rect2)
			if m == 0:
				plt.ylabel('Marginal Impact', fontsize = fs)
				if varsToDisplay == 'techCat':
					plt.gca().add_patch(rect3)
			elif m == 1 and varsToDisplay == 'techCat':
				plt.gca().add_patch(rect4)
			elif m == 2 and varsToDisplay == 'techCat':
				plt.gca().add_patch(rect5)
				plt.gca().add_patch(rect6)


		plt.savefig('Figures/'+figName+'.png', dpi= 300)
		plt.clf()

	makeBarFig('ppt', 'All', 'MarginalEffects_All')
	makeBarFig('ppt', 'OPEN', 'MarginalEffects_OPEN')
	makeBarFig('ppt', 'cohort', 'MarginalEffects_Cohort')
	makeBarFig('ppt', 'companyType', 'MarginalEffects_CompanyType')
	makeBarFig('ppt', 'techCat', 'MarginalEffects_TechCat')
	makeBarFig('ppt', 'awardAmount', 'MarginalEffects_AwardAmt')
#displayModelResults()


def plotOutcomeByYear(df):
	# make a plot of the different outcomes by year 
	plt.figure(figsize=(6,5))
	count_series = df.groupby(['startYr', 'FinalDecision']).size()
	new_df = count_series.to_frame(name = 'breakdown').reset_index()

	pivots = new_df[new_df.FinalDecision == 'Pivot']
	persists = new_df[new_df.FinalDecision == 'Persist']
	perishes = new_df[new_df.FinalDecision == 'Perish']

	ea = df.groupby(['early', 'FinalDecision']).size()[1]
	mi = df.groupby(['middle', 'FinalDecision']).size()[1]
	la = df.groupby(['late', 'FinalDecision']).size()[1]
	
	ax = 0.1
	lx = 0.18
	sx = 0.05
	ay = 0.15
	ly = 0.35
	sy = 0.1

	plt.subplot(position = [ax, ly+sy+ay, lx, ly])
	print('pivot by yr', pivots)
	plt.bar(pivots.startYr, pivots.breakdown)
	plt.title('Pivots', fontsize = 6)
	plt.xticks([0, 2, 4, 6, 8], ('2009', '2011', '2013', '2015', '2017'), fontsize = 6, rotation = 90)
	plt.yticks([0, 50, 100, 150, 200, 250], fontsize = 6)
	plt.ylim(0,250)
	plt.ylabel('Awards')

	plt.subplot(position = [ax+sx+lx, ly+sy+ay, lx, ly])
	plt.bar(persists.startYr, persists.breakdown)
	plt.title('Persists', fontsize = 6)
	plt.xticks([0, 2, 4, 6, 8], ('2009', '2011', '2013', '2015', '2017'), fontsize = 6, rotation = 90)
	plt.yticks([0, 50, 100, 150, 200, 250], labels = ' ', fontsize = 6)
	plt.ylim(0,250)

	plt.subplot(position = [ax+2*sx+2*lx, ly+sy+ay, lx, ly])
	plt.bar(perishes.startYr, perishes.breakdown)
	plt.title('Perishes', fontsize = 6)
	plt.xticks([0, 2, 4, 6, 8], ('2009', '2011', '2013', '2015', '2017'), fontsize = 6, rotation = 90)
	plt.yticks([0, 50, 100, 150, 200, 250], labels = ' ', fontsize = 6)
	plt.ylim(0,250)

	plt.subplot(position = [ax+3*sx+3*lx, ly+sy+ay, lx, ly])
	plt.bar(perishes.startYr, df.groupby(['startYr']).size())
	plt.title('Total Projects', fontsize = 6)
	plt.xticks([0, 2, 4, 6, 8], ('2009', '2011', '2013', '2015', '2017'), fontsize = 6, rotation = 90)
	plt.yticks([0, 50, 100, 150, 200, 250], labels = ' ', fontsize = 6)
	plt.ylim(0,250)

	plt.subplot(position = [ax, ay, lx, ly])
	plt.bar([1,2,3], [ea.loc['Pivot'], mi.loc['Pivot'], la.loc['Pivot']])
	#plt.title('Pivots', fontsize = 6)
	plt.xticks([1,2,3], ('2009-2011', '2012-2014', '2015-2017'), fontsize = 6, rotation = 45)
	plt.yticks([0, 50, 100, 150, 200, 250],  fontsize = 6)
	plt.ylim(0,250)
	plt.ylabel('Awards')

	plt.subplot(position = [ax+sx+lx, ay, lx, ly])
	plt.bar([1,2,3], [ea.loc['Persist'], mi.loc['Persist'], la.loc['Persist']])
	#plt.title('Persists', fontsize = 6)
	plt.ylim(0,250)
	plt.xticks([1,2,3], ('2009-2011', '2012-2014', '2015-2017'), fontsize = 6, rotation = 45)
	plt.yticks([0, 50, 100, 150, 200, 250], labels = ' ', fontsize = 6)

	plt.subplot(position = [ax+2*sx+2*lx, ay, lx, ly])
	plt.bar([1,2,3], [ea.loc['Perish'], mi.loc['Perish'], la.loc['Perish']])
	#plt.title('Perish', fontsize = 6)
	plt.ylim(0,250)
	plt.xticks([1,2,3], ('2009-2011', '2012-2014', '2015-2017'), fontsize = 6, rotation = 45)
	plt.yticks([0, 50, 100, 150, 200, 250], labels = ' ', fontsize = 6)
	plt.text(0, -90, 'Year', horizontalalignment = 'center')
	#plt.xlabel('Year')

	plt.subplot(position = [ax+3*sx+3*lx, ay, lx, ly])
	plt.bar([1,2,3], [df.groupby(['early']).size()[1], df.groupby(['middle']).size()[1], df.groupby(['late']).size()[1]])
	#plt.title('Total Projects', fontsize = 6)
	plt.ylim(0,250)
	plt.xticks([1,2,3], ('2009-2011', '2012-2014', '2015-2017'), fontsize = 6, rotation = 45)
	plt.yticks([0, 50, 100, 150, 200, 250], labels = ' ', fontsize = 6)

	#plt.bar(range(8), [new_df.breakdown[0], new_df.breakdown[3], new_df.breakdown[6], new_df.breakdown[9], new_df.breakdown[12], new_df.breakdown[15], new_df.breakdown[18], new_df.breakdown[19]])
	#plt.subplot(1,3,1)
	#pivot = df[df.FinalDecision == 'Pivot']

	#plt.hist(pivot.startYr, pivot.FinalDecision)
	plt.savefig('Figures/outcomeByYear.png', dpi=300)


	#plt.clf()
	print(count_series)
#plotOutcomeByYear(df)

def makeAMETables(modmg): #makes a csv version of the odds ratio 
	outdf = pd.DataFrame(columns = ('Parameter', 'Perish', 'Persist', 'Pivot'))
	outdf[0]['Parameter'] = modmg(['dy/dx'][('FinalDecision=Perish', 'OPEN')])
	print(outdf)

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
def makeFrequencyBarChart(df):
	print(df.columns)
	y1=(df.FinalDecision.value_counts())
	print(y1)
	subDf = df[df.techCat1 == 'Storage']
	y2 = (subDf.FinalDecision.value_counts())
	#print(df.FinalDecision)

	plt.figure(figsize=(4,4.75))
	plt.subplot(position = [0.15, 0.1, 0.85, 0.85])
	plt.bar(range(3), y1)
	plt.bar(range(3), y2, color = c1[1])
	plt.ylabel('Frequency')
	plt.xticks(range(3), ('Pivot', 'Perish', 'Persist'))
	plt.xlim(-0.5, 3.5)
	plt.text(2.45, 5, 'Storage')
	plt.text(2.45, 57, 'Other\nCategories')
	plt.savefig('Figures/outcomeStorage.png', dpi = 300)
#makeFrequencyBarChart(df)


d.makeSummaryByCategory(df, 'ppt')

#def dupeAnalysis(df):

