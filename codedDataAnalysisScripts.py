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
		df['endYr'] = 0
		df['startYr'] = 0
		df['yrGrp'] = 0
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

		df['ForProfStartup'] = 0
		df.loc[(df.CompanyType == 'Private, Start-Up'), 'ForProfStartup'] = 1
		df.loc[(df.CompanyType == 'Non-profit'), 'ForProfStartup'] = 2
		df['StartupForProf'] = 0
		df['OtherForProf'] = 1
		df['NonProf'] = 0
		df.loc[(df.CompanyType)=='Private, Start-Up', 'StartupForProf'] = 1
		df.loc[(df.CompanyType)=='Private, Start-Up', 'OtherForProf'] = 0
		df.loc[(df.CompanyType)=='Non-profit', 'OtherForProf'] = 0
		df.loc[(df.CompanyType)=='Non-profit', 'NonProf'] = 1

		df['SmallForProf'] = 0
		df['LargeForProf'] = 0
		df.loc[(df.Size)=='Small', 'SmallForProf'] = 1
		df.loc[(df.Size)=='Large', 'LargeForProf'] = 1

		df.loc[(df.FinalDecision=='Perish'), 'dumPerish'] = 1
		df.loc[(df.FinalDecision=='Pivot'), 'dumPivot'] = 1
		df.loc[(df.FinalDecision=='Persist'), 'dumPersist'] = 1

		df = df[(df.awardAmount!=0)]

		df = df.reset_index(drop = True)

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

		#df = df.reset_index(drop = True)
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

		count_series = df.groupby(['ForProfStartup', 'FinalDecision']).size()
		#print(count_series)
		new_df = count_series.to_frame(name = 'breakdown').reset_index()
		stat, p, dof, expected = stats.chi2_contingency([new_df.breakdown[0:3], new_df.breakdown[3:6], new_df.breakdown[6:9]])
		print('recipientType with startups', 'stat', stat, 'p', p, 'dof', dof, 'expected', expected)		

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

	def makeSummaryFigures(self, df, presType):
		if presType == 'paper':
			fgs = (3.33,4)
			fs = 7
		elif presType == 'ppt':
			fgs = (9, 4.5)
			fs = 14

		# bar chart with few outcomes
		def barChartPaper():
			plt.figure(figsize = fgs)
			plt.bar(1, sum(df.OPEN))
			plt.text(1, sum(df.OPEN)/2, ('OPEN\n' + str(sum(df.OPEN))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')
			plt.bar(1, sum(1-df.OPEN), bottom = sum(df.OPEN))
			plt.text(1, sum(df.OPEN)+sum(1-df.OPEN)/2, ('Designed\n' +  str(sum(1-df.OPEN))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')

			plt.bar(2, [sum(df.StartupForProf)])
			plt.text(2, sum(df.StartupForProf)/2, ('Startups\n' + str(sum(df.StartupForProf))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')
			
			plt.bar(2, sum(df.OtherForProf), bottom = sum(df.StartupForProf))
			plt.text(2, sum(df.StartupForProf)+sum(df.OtherForProf)/2, ('Other\nFor-\nProfit\n' + str(sum(df.OtherForProf))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')
			
			plt.bar(2, sum(df.NonProf), bottom = (sum(df.StartupForProf)+sum(df.OtherForProf)))
			plt.text(2, sum(df.OtherForProf)+ sum(df.StartupForProf)+sum(df.NonProf)/2, ('Non-\nProfits\n' + str(sum(df.NonProf))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')

			plt.bar(3, sum(df.SmallForProf))
			plt.text(3, sum(df.SmallForProf)/2, ('Small\nFor-\nProfits\n'+str(sum(df.SmallForProf))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')
			plt.bar(3, sum(df.LargeForProf), bottom = sum(df.SmallForProf))
			plt.text(3, sum(df.SmallForProf)+sum(df.LargeForProf)/2, ('Large\nFor-\nProfits\n'+str(sum(df.LargeForProf))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')
			plt.bar(3, sum(df.NonProf), bottom = sum(df.SmallForProf)+sum(df.LargeForProf))
			plt.text(3, sum(df.SmallForProf)+sum(df.LargeForProf)+sum(df.NonProf)/2, ('Non-\nProfits\n'+str(sum(df.NonProf))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')

			plt.bar(4, sum(df.Partners))
			plt.text(4, sum(df.Partners)/2, ('Partners\n' + str(sum(df.Partners))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')
			plt.bar(4, sum(1-df.Partners), bottom = sum(df.Partners))
			plt.text(4, sum(1-df.Partners)/2+sum(df.Partners), ('No\nPartners\n' + str(sum(1-df.Partners))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')

			plt.bar(5, sum(df.TC_TF))
			plt.text(5.5, sum(df.TC_TF)/2, ('Transportation\nFuels ' + str(sum(df.TC_TF))), fontsize = fs-1, verticalAlignment = 'center')
			b = sum(df.TC_TF)

			plt.bar(5, sum(df.TC_DG), bottom = b)
			plt.text(5.5, b + sum(df.TC_DG)/2, ('Distributed\nGeneration '+ str(sum(df.TC_DG))), fontsize =fs-1, verticalAlignment = 'center')
			b = b + sum(df.TC_DG)

			plt.bar(5, sum(df.TC_TS), bottom = b)
			plt.text(5.5, b+ sum(df.TC_TS)/2, ('Transportation\nStorage '+ str(sum(df.TC_TS))), fontsize = fs-1, verticalAlignment = 'center')
			b = b + sum(df.TC_TS)
			
			plt.bar(5, sum(df.TC_SS), bottom = b)
			plt.text(5.5, b + sum(df.TC_SS)/2, ('Stationary\nStorage '+ str(sum(df.TC_SS))), fontsize =fs-1, verticalAlignment = 'center')
			b = b + sum(df.TC_SS)

			plt.bar(5, sum(df.TC_BE), bottom = b)
			plt.text(5.5, b + sum(df.TC_BE)/2, ('Building\nEfficiency '+ str(sum(df.TC_BE))), fontsize =fs-1, verticalAlignment = 'center')
			b = b + sum(df.TC_BE)

			plt.bar(5, sum(df.TC_RE), bottom = b)
			plt.text(5.5, b + sum(df.TC_RE)/2, ('Resource\nEfficiency '+ str(sum(df.TC_RE))), fontsize =fs-1, verticalAlignment = 'center')
			b = b + sum(df.TC_RE)

			plt.bar(5, sum(df.TC_ME), bottom = b)
			plt.text(5.5, b + sum(df.TC_ME)/2, ('Mfg. Eff. '+ str(sum(df.TC_ME))), fontsize =fs-1, verticalAlignment = 'center')
			b = b + sum(df.TC_ME)

			plt.bar(5, sum(df.TC_EE), bottom = b)
			plt.text(5.5, b + sum(df.TC_EE)/2, ('Elec. Eff. '+ str(sum(df.TC_EE))), fontsize =fs-1, verticalAlignment = 'center')
			b = b + sum(df.TC_EE)
			
			plt.bar(5, sum(df.TC_GR), bottom = b)
			plt.text(5.5, b + sum(df.TC_GR)/2, ('Grid '+ str(sum(df.TC_GR))), fontsize =fs-1, verticalAlignment = 'center')
			b = b + sum(df.TC_GR)

			plt.bar(5, sum(df.TC_OT), bottom = b)
			plt.text(5.5, b + sum(df.TC_OT)/2, ('Other '+ str(sum(df.TC_OT))), fontsize =fs-1, verticalAlignment = 'center')
			b = b + sum(df.TC_OT)
			
			plt.xlim(0.5, 7.5)

			plt.savefig('Figures/SummaryBarChart.png', dpi = 300)

		def barChartPPT():
			plt.figure(figsize = fgs)
			plt.subplot(position = [0.17, 0.15, 0.8, 0.8])
			plt.barh(1, sum(df.OPEN))
			plt.text(sum(df.OPEN)/2, 1, ('OPEN\n' + str(sum(df.OPEN))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')
			plt.barh(1, sum(1-df.OPEN), left = sum(df.OPEN))
			plt.text(sum(df.OPEN)+sum(1-df.OPEN)/2, 1, ('Designed\n' +  str(sum(1-df.OPEN))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')

			plt.barh(2, [sum(df.StartupForProf)])
			plt.text(sum(df.StartupForProf)/2, 2, ('Startups\n' + str(sum(df.StartupForProf))), fontsize = fs-2, horizontalAlignment = 'center', verticalAlignment = 'center')
			
			plt.barh(2, sum(df.OtherForProf), left = sum(df.StartupForProf))
			plt.text(sum(df.StartupForProf)+sum(df.OtherForProf)/2, 2, ('Other For-Profit\n' + str(sum(df.OtherForProf))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')
			
			plt.barh(2, sum(df.NonProf), left = (sum(df.StartupForProf)+sum(df.OtherForProf)))
			plt.text(sum(df.OtherForProf)+ sum(df.StartupForProf)+sum(df.NonProf)/2, 2, ('Non-Profits\n' + str(sum(df.NonProf))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')

			plt.barh(3, sum(df.SmallForProf))
			plt.text(sum(df.SmallForProf)/2, 3, ('Small For-Profits\n'+str(sum(df.SmallForProf))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')
			plt.barh(3, sum(df.LargeForProf), left = sum(df.SmallForProf))
			plt.text(sum(df.SmallForProf)+sum(df.LargeForProf)/2, 3, ('Large For-\nProfits '+str(sum(df.LargeForProf))), fontsize = fs-2, horizontalAlignment = 'center', verticalAlignment = 'center')
			plt.barh(3, sum(df.NonProf), left = sum(df.SmallForProf)+sum(df.LargeForProf))
			plt.text(sum(df.SmallForProf)+sum(df.LargeForProf)+sum(df.NonProf)/2, 3, ('Non-Profits\n'+str(sum(df.NonProf))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')

			plt.barh(4, sum(df.Partners))
			plt.text(sum(df.Partners)/2, 4, ('Partners\n' + str(sum(df.Partners))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')
			plt.barh(4, sum(1-df.Partners), left = sum(df.Partners))
			plt.text(sum(1-df.Partners)/2+sum(df.Partners), 4, ('No Partners\n' + str(sum(1-df.Partners))), fontsize = fs, horizontalAlignment = 'center', verticalAlignment = 'center')

			plt.barh(5, sum(df.TC_TF))
			plt.text(sum(df.TC_TF)/2, 5, ('Transportation\nFuels ' + str(sum(df.TC_TF))), fontsize = fs-3, verticalAlignment = 'center', horizontalAlignment = 'center')
			b = sum(df.TC_TF)

			plt.barh(5, sum(df.TC_DG), left = b)
			plt.text(b + sum(df.TC_DG)/2, 5, ('Distributed\nGen. '+ str(sum(df.TC_DG))), fontsize =fs-3, verticalAlignment = 'center', horizontalAlignment = 'center')
			b = b + sum(df.TC_DG)

			plt.barh(5, sum(df.TC_TS), left = b)
			plt.text(b+ sum(df.TC_TS)/2, 5.6, ('Transportation\nStorage '+ str(sum(df.TC_TS))), fontsize = fs-3, verticalAlignment = 'bottom', horizontalAlignment = 'center')
			plt.plot([sum(df.TC_TS)/2 + b, sum(df.TC_TS)/2 + b], [5.4, 5.5], '-k')
			b = b + sum(df.TC_TS)
			
			plt.barh(5, sum(df.TC_SS), left = b)
			plt.text(b + sum(df.TC_SS)/2, 5, ('Stationary\nStorage '+ str(sum(df.TC_SS))), fontsize =fs-3, verticalAlignment = 'center', horizontalAlignment = 'center')
			b = b + sum(df.TC_SS)

			plt.barh(5, sum(df.TC_BE), left = b)
			plt.text(b + sum(df.TC_BE)/2, 5.6, ('Building\nEff. '+ str(sum(df.TC_BE))), fontsize =fs-3, verticalAlignment = 'bottom', horizontalAlignment = 'center')
			plt.plot([sum(df.TC_BE)/2 + b, sum(df.TC_BE)/2 + b], [5.4, 5.5], '-k')
			b = b + sum(df.TC_BE)

			plt.barh(5, sum(df.TC_RE), left = b)
			plt.text(b + sum(df.TC_RE)/2, 5, ('Resource\nEff. '+ str(sum(df.TC_RE))), fontsize =fs-3, verticalAlignment = 'center', horizontalAlignment = 'center')
			b = b + sum(df.TC_RE)

			plt.barh(5, sum(df.TC_ME), left = b)
			plt.text(b + sum(df.TC_ME)/2, 5.6, ('Mfg.\nEff. '+ str(sum(df.TC_ME))), fontsize =fs-3, verticalAlignment = 'bottom', horizontalAlignment = 'center')
			plt.plot([sum(df.TC_ME)/2 + b, sum(df.TC_ME)/2 + b], [5.4, 5.5], '-k')
			b = b + sum(df.TC_ME)

			plt.barh(5, sum(df.TC_EE), left = b)
			plt.text(b + sum(df.TC_EE)/2, 5, ('Elec.\nEff. '+ str(sum(df.TC_EE))), fontsize =fs-3, verticalAlignment = 'center', horizontalAlignment = 'center')
			b = b + sum(df.TC_EE)
			
			plt.barh(5, sum(df.TC_GR), left = b)
			plt.text(b + sum(df.TC_GR)/2, 5, ('Grid\n'+ str(sum(df.TC_GR))), fontsize =fs-3, verticalAlignment = 'center', horizontalAlignment = 'center')
			b = b + sum(df.TC_GR)

			plt.barh(5, sum(df.TC_OT), left = b)
			plt.text(b + sum(df.TC_OT)/2, 5.6, ('Other\n'+ str(sum(df.TC_OT))), fontsize =fs-3, verticalAlignment = 'bottom', horizontalAlignment = 'center')
			plt.plot([sum(df.TC_OT)/2 + b, sum(df.TC_OT)/2 + b], [5.4, 5.5], '-k')
			b = b + sum(df.TC_OT)
			
			plt.ylim(0.5, 6.5)
			plt.yticks([1,2,3,4,5], ('OPEN', 'Startups', 'For Profit Size', 'Partners', 'Tech Category'), fontsize = fs)
			plt.xlim(0, 500)
			plt.xticks([0, 100, 200, 300, 400], fontsize = fs)
			plt.xlabel('Projects', fontsize = fs)

			plt.savefig('Figures/SummaryBarChart.png', dpi = 300)	

		if presType == 'paper':
			barChartPaper()
		else:
			barChartPPT()
		
		# award histograms
		plt.figure(figsize = fgs)
		plt.subplot(position = [0.1, 0.2, 0.18, 0.7])
		#print(max(df.awardAmount))
		x = df[df.FinalDecision == 'Persist']
		plt.hist(x.awardAmount, bins = [0,1,2,3,4,5,6,7,8,9,10])
		plt.ylim(0, 125)
		plt.ylabel('Number of Awards')
		plt.title('Persist')

		plt.subplot(position = [0.33, 0.2, 0.18, 0.7])
		#print(max(df.awardAmount))
		x = df[df.FinalDecision == 'Pivot']
		plt.hist(x.awardAmount, bins = [0,1,2,3,4,5,6,7,8,9,10])
		plt.ylim(0, 125)
		plt.text(15, -20, 'Award Amount', horizontalAlignment = 'center')
		plt.title('Pivot')

		plt.subplot(position = [0.56, 0.2, 0.18, 0.7])
		#print(max(df.awardAmount))
		x = df[df.FinalDecision == 'Perish']
		plt.hist(x.awardAmount, bins = [0,1,2,3,4,5,6,7,8,9,10])
		plt.ylim(0, 125)
		plt.title('Perish')

		plt.subplot(position = [0.79, 0.2, 0.18, 0.7])
		plt.hist(df.awardAmount, bins = [0,1,2,3,4,5,6,7,8,9,10])
		plt.ylim(0, 125)
		plt.title('Total Projects')
		plt.savefig('Figures/Histogram_awardAmts.png', dpi = 300)
		plt.clf()

		# projects by profit status
		plt.figure(figsize = fgs)
		plt.pie([sum(df.ForProf), sum(df.NonProf)], labels = ['For-profit', 'Non-profit'], autopct='%1.1f%%')
		plt.savefig('Figures/PieChart_ForProf.png', dpi=300)
		plt.clf()

		# projects by company type (startup)
		plt.figure(figsize=fgs)
		print(sum(df.StartupForProf), sum(df.OtherForProf), sum(df.NonProf))
		plt.pie([sum(df.StartupForProf), sum(df.OtherForProf), sum(df.NonProf)], labels = ['Startup', 'Other For Profit', 'Non-profit'], autopct='%1.1f%%')
		plt.savefig('Figures/PieChart_Startup.png', dpi=300)
		plt.clf()

		# projects by company type (large/small forprof)
		plt.figure(figsize = fgs)
		print(sum(df.LargeForProf), sum(df.SmallForProf), sum(df.NonProf))
		plt.pie([sum(df.LargeForProf), sum(df.SmallForProf), sum(df.NonProf)], labels = ['Large\nFor-\nprofit', 'Small\nFor Profit', 'Non-profit'], autopct='%1.1f%%')
		plt.savefig('Figures/PieChart_Size.png', dpi=300)
		plt.clf()

		# outcomes
		plt.figure(figsize = fgs)
		s = df.groupby('FinalDecision').size()
		print(s)
		plt.pie(s, labels = ['Perish','Persist', 'Pivot'], autopct='%1.1f%%')
		plt.savefig('Figures/PieChart_Outcome.png', dpi= 300)
		plt.clf()

		# projects by tech category & outcome
		plt.figure(figsize = fgs)
		ax1 = plt.subplot(1,3,1)
		ax2 = plt.subplot(1,3,2)
		ax3 = plt.subplot(1,3,3)
		ex_list = ['TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT']
		for n in range(len(ex_list)):
			s1 = len(df[(df.FinalDecision=='Persist') & (df[ex_list[n]]==1)])
			s2 = len(df[(df.FinalDecision=='Pivot') & (df[ex_list[n]]==1)])
			s3 = len(df[(df.FinalDecision=='Perish') & (df[ex_list[n]]==1)])
			print(s1,s2,s3)

			ax1.bar(n, s1)
			ax2.bar(n, s2)
			ax3.bar(n, s3)
		ax1.set_ylim(0,40)
		ax2.set_ylim(0,40)
		ax3.set_ylim(0,40)
		ax1.set_xticks(range(len(ex_list)))
		ax1.set_xticklabels(ex_list, fontsize = fs, rotation = 90)
		ax2.set_yticklabels('')
		ax3.set_yticklabels('')
		ax1.set_title('Persist')
		ax2.set_title('Pivot')
		ax3.set_title('Perish')
		plt.savefig('Figures/TestBar.png', dpi=300)
		print(df.shape)

	def makeSummaryByCategory(self, df, presType):
		# by tech category
		ex_list = ['TC_SS', 'TC_TF', 'TC_DG', 'TC_TS', 'TC_BE', 'TC_RE', 'TC_ME', 'TC_EE', 'TC_GR', 'TC_OT']
		labels = ('Storage', 'Trans Fuels', 'Dist Gen', 'Trans Storage', 'Bdlg Eff', 'Resource Eff', 'Mfg Eff', 'Elec Eff', 'Grid', 'Other')
		x_plot = np.empty((10,4))
		
		for n in range(len(ex_list)):
			x = df.groupby([ex_list[n], 'FinalDecision']).size()
			x_plot[n,0] = x[1].loc['Perish']
			x_plot[n,1] = x[1].loc['Persist']
			x_plot[n,2] = x[1].loc['Pivot']
		x_plot[:,3] = np.sum(x_plot[:,0:3], axis = 1)
		if presType == 'paper':
			fgs = (3.33,4)
			fs = 7
		elif presType == 'ppt':
			fgs = (9, 4.5)
			fs = 14
		
		plt.figure(figsize = fgs)
		plt.subplot(position = [0.17, 0.15, 0.8, 0.8])
		plt.barh(range(len(ex_list)), x_plot[:,0])
		plt.barh(range(len(ex_list)), x_plot[:,1], left = x_plot[:,0])
		plt.barh(range(len(ex_list)), x_plot[:,2], left = x_plot[:,0]+x_plot[:,1])

		for n in range(len(ex_list)):
			plt.text(x_plot[n,0]/2, n, "%0.1f" % (100*x_plot[n,0]/x_plot[n,3]) + '%', horizontalAlignment = 'center', verticalAlignment = 'center')
			plt.text(x_plot[n,0]+x_plot[n,1]/2, n, "%0.1f" % (100*x_plot[n,1]/x_plot[n,3]) + '%', horizontalAlignment = 'center', verticalAlignment = 'center')
			plt.text(x_plot[n,0]+x_plot[n,1]+x_plot[n,2]/2, n, "%0.1f" % (100*x_plot[n,2]/x_plot[n,3]) + '%', horizontalAlignment = 'center', verticalAlignment = 'center')
		plt.legend(('Perish', 'Persist', 'Pivot'), fontsize = fs)
		plt.xlabel('Number of Projects', fontsize = fs)
		plt.xlim(0,80)
		plt.xticks([0,10,20,30,40,50,60,70], fontsize =fs)
		plt.yticks(range(len(ex_list)), labels = labels, fontsize = fs)

		plt.savefig('Figures/OutcomeByTC_Bar.png', dpi = 300)

		
		x = df.groupby('PD').size()
		y1 = x.where(x>5).dropna()
		y2 = x.where(x<6).dropna()
		df['PD_edited']=""

		for n in range(len(df)):
			if df.PD[n] in y1:
				df.PD_edited[n] = df.PD[n]
			elif df.PD[n] in y2:
				df.PD_edited[n] = 'Other'

		z = df.groupby(['PD_edited', 'FinalDecision']).size()
		z2 = df.PD_edited.unique()
		
		y_plot = np.empty((len(z2)+1, 4))
		for n in range(len(z2)):
			x = z.loc[z2[n]]
			if 'Perish' in x:
				y_plot[n, 0] = x.loc['Perish']
			else:
				y_plot[n, 0] = 0
			if 'Persist' in  x:
				y_plot[n,1] = x.loc['Persist']
			else:
				y_plot[n, 1] = 0

			if 'Pivot' in x:
				y_plot[n,2] = x.loc['Pivot']
			else:
				y_plot[n, 2] = 0

		y_plot[:,3] = np.sum(y_plot[:,0:3], axis = 1)
		y_plot[-1,:] = np.sum(y_plot[0:-1,:], axis = 0)


		plt.figure(figsize = fgs)
		plt.subplot(position = [0.08, 0.15, 0.9, 0.8])
		
		plt.barh(range(len(z2)), y_plot[0:-1,0])
		plt.barh(range(len(z2)), y_plot[0:-1,1], left = y_plot[0:-1,0])
		plt.barh(range(len(z2)), y_plot[0:-1,2], left = y_plot[0:-1,0]+y_plot[0:-1,1])

		for n in range(len(z2)):
			if y_plot[n,0]>0:
				plt.text(y_plot[n,0]/2, n, "%0.0f" % (100*y_plot[n,0]/y_plot[n,3]) + '%', horizontalAlignment = 'center', verticalAlignment = 'center', fontsize = 6)
			if y_plot[n,1]>0:
				plt.text(y_plot[n,0]+y_plot[n,1]/2, n, "%0.0f" % (100*y_plot[n,1]/y_plot[n,3]) + '%', horizontalAlignment = 'center', verticalAlignment = 'center', fontsize = 6)
			plt.text(y_plot[n,0]+y_plot[n,1]+y_plot[n,2]/2, n, "%0.0f" % (100*y_plot[n,2]/y_plot[n,3]) + '%', horizontalAlignment = 'center', verticalAlignment = 'center', fontsize = 6)
		plt.legend(('Perish', 'Persist', 'Pivot'), fontsize = fs)
		plt.xlabel('Number of Projects', fontsize = fs)
		plt.xlim(0,50)
		plt.ylim(-1, 27)
		plt.xticks([0,10,20,30,40], fontsize =fs)
		plt.yticks(range(len(z2)), labels = range(1, len(z2)+1), fontsize = fs-4)
		plt.text(30, 5, '* Combined')
		plt.ylabel('Program Director', fontsize = fs)

		plt.savefig('Figures/OutcomeByPD_Bar.png', dpi = 300)
		
		z_plot = np.empty((len(ex_list),2))
		for n in range(len(ex_list)):
			x = df.groupby([ex_list[n], 'PD']).size()
			z_plot[n,0] = len(x[1])
			x = df.groupby([ex_list[n]]).size()
			z_plot[n,1] = x[1]
		
		plt.figure(figsize = fgs)
		plt.subplot(position = [0.1, 0.25, 0.8, 0.73])
		plt.bar(range(len(ex_list)), z_plot[:,1])
		plt.bar(range(len(ex_list)), z_plot[:,0])
		plt.plot(range(len(ex_list)), z_plot[:,1]/z_plot[:,0], '.k')
		for n in range(len(ex_list)):
			plt.text(n, z_plot[n,1]+0.5, "%0.1f" % (z_plot[n,1]/z_plot[n,0]), horizontalAlignment = 'center')
			plt.text(n, z_plot[n,0]+0.5, "%0.0f" % z_plot[n,0], horizontalAlignment = 'center', color = '#ff7f0e')
		plt.xticks(range(len(ex_list)), labels = labels, fontsize = fs-4, rotation = 90)
		plt.legend(('Ratio (Projects:PD)', 'Projects', 'Program Directors'))
		plt.savefig('Figures/PDsbyTechCat.png', dpi = 300)
		plt.clf()


		dd = df[(df.TC_TF == 1)|(df.TC_TS == 1)|(df.TC_DG == 1)]
		pds = dd.PD.unique().tolist()
		plotData = np.zeros((len(pds), 3, 3))
		x = df[df.TC_TF==1]
		x1 = x.groupby(['FinalDecision', 'PD']).size()
		pdsx = x.PD.unique().tolist()
		for n in range(len(pdsx)):
			m = pds.index(pdsx[n])
			if pdsx[n] in x1.loc['Perish']:
				plotData[m,0,0] = x1.loc['Perish'].loc[pdsx[n]]
			else:
				plotData[m,0,0] = 0
			if pdsx[n] in x1.loc['Persist']:
				plotData[m,1,0] = x1.loc['Persist'].loc[pdsx[n]]
			else:
				plotData[m,1,0] = 0
			if pdsx[n] in x1.loc['Pivot']:
				plotData[m,2,0] = x1.loc['Pivot'].loc[pdsx[n]]
			else:
				plotData[m,2,0] = 0

		y = df[df.TC_TS==1]
		y1 = y.groupby(['FinalDecision', 'PD']).size()
		pdsy = y.PD.unique()


		for n in range(len(pdsy)):
			m = pds.index(pdsy[n])
			if pdsy[n] in y1.loc['Perish']:
				plotData[m,0,1] = y1.loc['Perish'].loc[pdsy[n]]
			else:
				plotData[m,0,1] = 0
			if pdsy[n] in y1.loc['Persist']:
				plotData[m,1,1] = y1.loc['Persist'].loc[pdsy[n]]
			else:
				plotData[m,1,1] = 0
			if pdsy[n] in y1.loc['Pivot']:
				plotData[m,2,1] = y1.loc['Pivot'].loc[pdsy[n]]
			else:
				plotData[m,2,1] = 0

		z = df[df.TC_DG==1]
		z1 = z.groupby(['FinalDecision', 'PD']).size()
		pdsz = z.PD.unique()
		for n in range(len(pdsz)):
			m = pds.index(pdsz[n])
			if pdsz[n] in z1.loc['Perish']:
				plotData[m,0,2] = z1.loc['Perish'].loc[pdsz[n]]
			else:
				plotData[m,0,2] = 0
			if pdsz[n] in z1.loc['Persist']:
				plotData[m,1,2] = z1.loc['Persist'].loc[pdsz[n]]
			else:
				plotData[m,1,2] = 0
			if pdsz[n] in z1.loc['Pivot']:
				plotData[m,2,2] = z1.loc['Pivot'].loc[pdsz[n]]
			else:
				plotData[m,2,2] = 0


		pl = [[0.1, 0.1, 0.25, 0.8], [0.4, 0.1, 0.25, 0.8], [0.7, 0.1, 0.25, 0.8]]
		titles = ('Transportation Fuel', 'Transportation Storage', 'Distributed Generation')
		plt.figure(figsize = fgs)
		for q in range(3):
			plt.subplot(position = pl[q])
			plt.barh(range(len(pds)), plotData[:,0,q])
			plt.barh(range(len(pds)), plotData[:,1,q], left = plotData[:,0,q])
			plt.barh(range(len(pds)), plotData[:,2,q], left = plotData[:,0,q]+plotData[:,1,q])
			plt.xlim(0, 16)
			plt.yticks(range(len(pds)), labels = range(1, len(pds)+1), fontsize = 4)
			plt.title(titles[q])
			plt.legend(('Perish', 'Persist', 'Pivot'), fontsize = 7)
			if q == 0:
				plt.ylabel('Program Director')

		plt.savefig('Figures/PDsFor3Cats.png', dpi = 300)
		#y = df.groupby(['TC_TS', 'PD']).size()
		#print(y[1])
		#z = df.groupby(['TC_DG', 'PD']).size()
		#print(z[1])


		#x = df.groupby(['TC_SS', 'PD']).size()
		#print(len(x[1]))
		#y = df.groupby(['TC_TF', 'PD']).size()
		#print(len(y[1]))




