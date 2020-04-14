import numpy as np 
import pandas as pd 
#pd.set_option('display.max_columns', 10)
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
from scrips import chi2calc


class codedDataAnalysisScripts:

	def __init__(self,name):
		self.user = name

	def cleanData(self, df):
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

		#print(df.recipientType.value_counts())

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

	def univariateCategorical_2Categories(self, df):
		# calculate the chi2 based on open/designed outcomes 
		stat, p, dof, expected = chi2calc(df, 'OPEN', 'FinalDecision')
		print('OPEN', 'stat', stat, 'p', p, 'dof', dof, 'expected', expected)

		stat, p, dof, expected = chi2calc(df, 'recipientType', 'FinalDecision')
		print('recipientType', 'stat', stat, 'p', p, 'dof', dof, 'expected', expected)

		stat, p, dof, expected = chi2calc(df, 'Partners', 'FinalDecision')
		print('Partners', 'stat', stat, 'p', p, 'dof', dof, 'expected', expected)

	def univariateCategorical_multipleCategories(self, df):
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
		#print(new_df)
		stat, p, dof, expected = stats.chi2_contingency([new_df.breakdown[0:3], new_df.breakdown[3:6], new_df.breakdown[6:9], new_df.breakdown[9:12], new_df.breakdown[12:15], new_df.breakdown[15:18], new_df.breakdown[18:21], new_df.breakdown[21:24], new_df.breakdown[24:27]])# ddof = 2)
		print('start year', 'stat', stat, 'p', p, 'dof', dof, 'expected', expected)

		count_series = df.groupby(['yrGrp', 'FinalDecision']).size()
		new_df = count_series.to_frame(name = 'breakdown').reset_index()
		stat, p, dof, expected = stats.chi2_contingency([new_df.breakdown[0:3], new_df.breakdown[3:6], new_df.breakdown[6:9]])
		print('start year - grouped', 'stat', stat, 'p', p, 'dof', dof, 'expected', expected)

	def univariateContinuous(self, df):
		exog = df[['awardAmount']] 
		awardAmttest = sm.MNLogit(df.FinalDecision, exog.astype(float)).fit(maxiter = 10000, full_output = True)# method = 'bfgs')
		print(awardAmttest.summary())

	def modelSelectionComparison(self, mod1, mod2, filename):
		# do a likelihood ratio test 
		# figure out which model is bigger
		if len(mod1.params)>len(mod2.params):
			LLR = 2*(mod1.llf-mod2.llf)
			dfchange = len(mod1.params) - len(mod2.params)
			merge1 = pd.merge(left = mod1.params, right = mod2.params, left_index = True, right_index=True)
			merge1['big_pvals_persist'] = mod1.pvalues[0]
			merge1['small_pvals_persist'] = mod2.pvalues[0]
			merge1['big_pvals_pivot'] = mod1.pvalues[1]
			merge1['small_pvals_pivot'] = mod2.pvalues[1]
		else: 
			LLR = 2*(mod2.llf-mod1.llf)
			dfchange = len(mod2.params) - len(mod1.params)
			merge1 = pd.merge(left = mod2.params, right = mod1.params, left_index = True, right_index=True)
			merge1['big_pvals_persist'] = mod2.pvalues[0]
			merge1['small_pvals_persist'] = mod1.pvalues[0]
			merge1['big_pvals_pivot'] = mod2.pvalues[1]
			merge1['small_pvals_pivot'] = mod1.pvalues[1]
		a = stats.chisqprob(LLR, dfchange)
		print('Likelihood ratio test', LLR, 'p-value', a)
		merge1 = merge1.rename(columns={'0_x': 'big_param_persist', '1_x': 'big_param_pivot', '0_y': 'small_param_persist', '1_y':'small_param_pivot'})

		# compare the betas 
		merge1['persistComps'] = ""
		merge1['pivotComps'] = ""

		for n in range(len(merge1)):
			merge1.persistComps[n] = 100*(merge1.small_param_persist[n]-merge1.big_param_persist[n])/merge1.big_param_persist[n]
			merge1.pivotComps[n] = 100*(merge1.small_param_pivot[n]-merge1.big_param_pivot[n])/merge1.big_param_pivot[n]

		# reorder columns

		merge1 = merge1[['big_param_persist', 'small_param_persist', 'persistComps', 'big_pvals_persist', 'small_pvals_persist', 
		'big_param_pivot', 'small_param_pivot', 'pivotComps', 'big_pvals_pivot', 'small_pvals_pivot']]
		# save the comaprison of betas with significance data 

		merge1.to_csv(filename)


