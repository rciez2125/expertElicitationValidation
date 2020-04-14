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


# load the final compiled dataset
df = pd.read_csv('Data/FinalData_cleaned.csv')
d = codedDataAnalysisScripts.codedDataAnalysisScripts('a')

# do some data cleaning
df = d.cleanData(df)

## 1) univariable analysis of each independent variable 
#d.univariateCategorical_2Categories(df)
# nonstandard number of categories
#d.univariateCategorical_multipleCategories(df)
# continous variable test 
#d.univariateContinuous(df)

# based on this analysis, which variables matter?
# OPEN 						No
# Partners 					No
# Tech Category 			Yes
# Recipinet Type (binary)	Yes
# Recipient type (3)		Yes
# Start year (every year)	Yes
# Start year (grouped)		Yes
# Award Amount 				Yes 

### 2) Fit a multivariate model with variables identified in step 1
# exclude open, partners
# optionsf on recipient type and start year


def makeModelStep2(time, forProfType):
	ex_list = ['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT']

	if forProfType == 'three':
		ex_list.append('SmallForProf')
		ex_list.append('LargeForProf')
	else: 
		ex_list.append('ForProf')

	ex_list.append('awardAmount')

	if time == 'every year':
		ex_list.append('dum09')
		ex_list.append('dum10') 
		ex_list.append('dum11')
		ex_list.append('dum12')
		ex_list.append('dum13')
		ex_list.append('dum14') 
		ex_list.append('dum15')
		ex_list.append('dum16')
	elif time == 'grouped':
		ex_list.append('early')
		ex_list.append('middle')

	#ex_list.append('awardAmount')

	#print(ex_list)
	exog = df[ex_list]	
	mod = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)
	print(mod.summary())
	modmg = mod.get_margeff(at = 'overall')
	return(mod)

m1 = makeModelStep2('every year', 'three')
m2 = makeModelStep2('grouped', 'three')

print('Model 3')
m3 = makeModelStep2('every year', 'two')

print('Model 4')
m4 = makeModelStep2('grouped', 'two')
m5 = makeModelStep2('none', 'three')

print('Model 6')
m6 = makeModelStep2('none', 'two')


#print(m1.llr_pvalue)
#print(len(m1.params))

#d.modelSelectionComparison(m1, m2, 'All_3TvsGrouped_3T.csv')
#d.modelSelectionComparison(m1, m3, 'All_3TvsAll_2T.csv')
#d.modelSelectionComparison(m2, m4, 'Grouped_3TvsGrouped_2T.csv')
#d.modelSelectionComparison(m5, m6, 'None_3TvsNone_2T.csv')
d.modelSelectionComparison(m3, m4, 'All_2TvsGrouped_2T.csv')
d.modelSelectionComparison(m4, m6, 'Grouped_2TvsNone_2T.csv')
d.modelSelectionComparison(m3, m6, 'All_2Tvs_None_2T.csv')


# add other parameters back to model 4
def addOneVariable(time, variableToAdd):
	ex_list = ['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT', 'ForProf', 'awardAmount']

	if time == 'every year':
		ex_list.append('dum09')
		ex_list.append('dum10') 
		ex_list.append('dum11')
		ex_list.append('dum12')
		ex_list.append('dum13')
		ex_list.append('dum14') 
		ex_list.append('dum15')
		ex_list.append('dum16')
	elif time == 'grouped':
		ex_list.append('early')
		ex_list.append('middle')

	ex_list.append(variableToAdd)
	exog = df[ex_list]	
	mod = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)
	print(mod.summary())
	modmg = mod.get_margeff(at = 'overall')
	return(mod)

# try OPEN and parnters 

m7 = addOneVariable('every year', 'OPEN')
m8 = addOneVariable('grouped', 'OPEN')
m9 = addOneVariable('none', 'OPEN')
d.modelSelectionComparison(m3, m7, 'All2T_noOpenvsAll2T_Open.csv') # doesn't seem to help
d.modelSelectionComparison(m4, m8, 'Grouped2T_noOpenvsGrouped2T_Open.csv') #doesn't seem to help
d.modelSelectionComparison(m6, m9, 'None2T_noOpenvsNone2T_Open.csv') # doesn't seem to help

m10 = addOneVariable('every year', 'Partners')
m11 = addOneVariable('grouped', 'Partners')
m12 = addOneVariable('none', 'Partners')
d.modelSelectionComparison(m3, m10, 'All2T_noPartvsAll2T_Part.csv') # doesn't seem to help
d.modelSelectionComparison(m4, m11, 'Grouped2T_noPartvsGrouped2T_Part.csv') #doesn't seem to help
d.modelSelectionComparison(m6, m12, 'None2T_noPartvsNone2T_Part.csv') # doesn't seem to help, but is closer than other choices (p = 0.14)


# for 4/15:
# look at avg marginal effects of different time models --> are they substantially different?
# I don't think I can do an easy interpretation if I've got time in the models --> confirm this


# check for interactions 




# old stuff, mostly making plots

def plotOutcomeByYear(df):
	# make a plot of the different outcomes by year 
	plt.figure(figsize=(5,4))
	count_series = df.groupby(['startYr', 'FinalDecision']).size()
	new_df = count_series.to_frame(name = 'breakdown').reset_index()
	#pd.groupby.DataFrameGroupBy.plot(df, x='startYr', y='FinalDecision')
	#df.groupby['startYr', 'FinalDecision'].plot(kind='bar')

	pivots = new_df[new_df.FinalDecision == 'Pivot']
	persists = new_df[new_df.FinalDecision == 'Persist']
	perishes = new_df[new_df.FinalDecision == 'Perish']

	
	plt.subplot(1,3,1)
	plt.bar(pivots.startYr, pivots.breakdown)
	plt.title('Pivots')
	plt.ylim(0,60)

	plt.subplot(1,3,2)
	plt.bar(persists.startYr, persists.breakdown)
	plt.title('Persists')
	plt.ylim(0,60)

	plt.subplot(1,3,3)
	plt.bar(perishes.startYr, perishes.breakdown)
	plt.title('Perishes')
	plt.ylim(0,60)

	#plt.bar(range(8), [new_df.breakdown[0], new_df.breakdown[3], new_df.breakdown[6], new_df.breakdown[9], new_df.breakdown[12], new_df.breakdown[15], new_df.breakdown[18], new_df.breakdown[19]])
	#plt.subplot(1,3,1)
	#pivot = df[df.FinalDecision == 'Pivot']

	#plt.hist(pivot.startYr, pivot.FinalDecision)
	plt.savefig('Figures/outcomeByYear.png', dpi=300)
	#plt.clf()
	print(count_series)
plotOutcomeByYear(df)


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

#x1 = math.factorial(460)/(math.factorial(sum(df.dumPersist))*math.factorial(sum(df.dumPivot))*math.factorial(sum(df.dumPerish)))
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

#betastart = np.random.rand(3)-0.5
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

#betastart = np.random.rand(27)-0.5
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
	plt.savefig('Figures/outcomeStorage.png', dpi = 300)
#makeFrequencyBarChart(df)
