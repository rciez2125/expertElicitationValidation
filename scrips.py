from bs4 import BeautifulSoup
import requests
import time 
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from datetime import datetime

def scrapeSummaryData(page_link):
	page_response = requests.get(page_link)
	page_content = BeautifulSoup(page_response.content, "html.parser")
	holder = page_content.find_all(class_='field field-name-field-organization field-type-text field-label-hidden')
	companies = []
	for x in holder:
		a = str(x)
		b = a[138:]
		c = b.strip( '</div>' )
		companies.append(c)
	
	holder = page_content.find_all(class_='field field-name-title field-type-ds field-label-hidden')
	taglines = []
	projecturl = []
	for x in holder:
		a = str(x)
		b = a[146:]
		c = b.strip( '</div>' ) 
		d = c.strip( '</h2>' )   
		f = d.strip( '</a>' )
		h = f.rfind('>')+1
		k = f[h:]
		taglines.append(k)

		z = a.rfind('href')+6
		y = a[z:]
		w = y.strip( '</div>' ) 
		v = w.strip( '</h2>' )   
		u = v.strip( '</a>' )
		t = u.rfind('>')-1
		s = u[:t]
		projecturl.append(s)
	holder = page_content.find_all(class_='field field-name-field-program field-type-taxonomy-term-reference field-label-inline clearfix')
	program = []
	for x in holder:
		a = str(x)
		b = a.rfind('href')
		c = a[b:]
		d = c.strip( '</div>' ) 
		f = d.strip( '</a' )
		h = f.rfind('>')+1
		k = f[h:]
		program.append(k)
	holder = page_content.find_all(class_ = 'field field-name-field-project-start field-type-datetime field-label-inline clearfix')
	project_term1 = []
	project_term2 = []
	for x in holder:
		a = str(x)
		b = a[341:]
		c = b[:10]
		d = b.strip( '</div>' )
		f = d.strip( '</span>')
		g = f[-10:]
		project_term1.append(c)
		project_term2.append(g)
	holder = page_content.find_all(class_='field field-name-field-project-status field-type-text field-label-inline clearfix')
	project_status = []
	for x in holder:
		a = str(x)
		b = a[195:]
		c = b.strip( '</div>' )
		project_status.append(c)
	holder = page_content.find_all(class_='field field-name-field-project-state field-type-list-text field-label-inline clearfix')
	project_state = []
	for x in holder:
		a = str(x)
		b = a[198:]
		c = b.strip( '</div>' )
		project_state.append(c)
	#holder = page_content.find_all(class_='field field-name-field-innovation-advantages field-type-text-long field-label-hidden')
	holder = page_content.find_all(class_='panel-panel panel-col-middle')
	project_description = []
	for x in holder:
		a = str(x)
		b = a.find('field-item even')+16
		c = a[b:]
		d = c.strip('<p>')
		f = d.replace('</p>','')
		g = f.replace('</div>','')
		h = g.replace('</div', '')
		project_description.append(h)
	holder = page_content.find_all(class_='panel-panel panel-col-last')
	techCategories1 = []
	techCategories2 = []
	for x in range(len(holder)):
		if x%2 == 0:
			a = str(holder[x])
			d = a.find('skos:Concept')
			f = a[d+14:]
			g = f.find('skos:Concept')
			h = f.find('</a>')
			k = f[:h]
			m = f[g+14:]
			n = m.find('</a>')
			p = m[:n]
			techCategories1.append(p)
			q = m.find('skos:Concept')
			if q > 0:
				r = m[q+14:]
				s = r.find('</a>')
				t = r[:s]
				techCategories2.append(t)
			else:
				techCategories2.append(' ')


	dataOut = pd.DataFrame({'companies':companies, 'tagline':taglines, 
		'program':program, 'startDate':project_term1, 'endDate':project_term2, 
		'projectStatus':project_status, 'state':project_state,'description':project_description, 
		'techCat1':techCategories1, 'techCat2':techCategories2, 'projecturl':projecturl})
	return(dataOut)

def scrapeFundingData(inputData):
	# read the csv file 
	d = pd.read_csv(inputData)
	awardAmount = []
	for x in range(len(d.projecturl)):
		# append the url to be the full version
		u = 'https://arpa-e.energy.gov/' + d.projecturl[x]
		page_response = requests.get(u)
		page_content = BeautifulSoup(page_response.content, "html.parser")
		# find the award amount data
		a = page_content.find(class_='field field-name-field-arpae-award field-type-text field-label-inline clearfix')
		b = str(a)
		c = b.find('$')+1
		f = b[c:]
		g = f.replace('</div>','')
		print(g)
		# remove commas 
		for char in g:
			g = g.replace(",", "")	
		# save the award amount data
		print(g)
		awardAmount.append(g)
		print(x)
		time.sleep(1)
	# append the dataframe to include award amount
	d['awardAmount'] = awardAmount
	# return the dataframe
	return(d)

def plotByDate(data):
	# this will eventually do some things?
	print('Hello World')

def awardSizeHistogram(df):
	data = np.zeros(len(df.awardAmount))
	for x in range(len(df.awardAmount)):
		a = str(df.awardAmount[x])
		b = a.replace('$','')
		c = b.replace(',','')
		data[x] = c
	data = data/1000000
	openData = df.OPEN*data
	openData = openData[openData != 0] 
	#openData = np.append(openData, 0)
	openData = np.append(openData, 0)
	programData = abs(df.OPEN-1)*data 
	programData = programData[programData != 0]

	meanOpen = np.mean(openData)
	meanDesigned = np.mean(programData)
	sdOpen = np.std(openData)
	sdDesigned = np.std(programData)
	summaryStats = pd.DataFrame({'meanOpen':meanOpen, 'sdOpen':sdOpen, 'meanDesigned':meanDesigned, 'sdDesigned':sdDesigned}, index = [0])

	plt.figure(figsize=(5,4))
	plt.subplot(1,2,1)
	plt.hist(openData, bins=np.linspace(0, 10, 20))
	plt.ylim(0,80)
	plt.text(5, 75, 'n =' + str(len(openData)), horizontalAlignment='center', fontsize = 8)
	plt.text(5, 70, 'mean = ' + str(round(meanOpen, 2)), horizontalAlignment = 'center', fontsize = 8)
	plt.text(5, 65, 'sd =' + str(round(sdOpen,2)), horizontalAlignment = 'center', fontsize = 8)
	plt.ylabel('Number of Awards', fontsize = 8)
	plt.xlabel('Award Amount (Millions of $)', fontsize = 8)
	plt.title('OPEN Projects', fontsize = 8)

	plt.subplot(1,2,2)
	plt.hist(programData[programData !=0], bins=np.linspace(0, 10, 20))
	plt.ylim(0,80)
	plt.text(5, 75, 'n =' + str(len(programData)), horizontalAlignment='center', fontsize = 8)
	plt.text(5, 70, 'mean = ' + str(round(meanDesigned, 2)), horizontalAlignment = 'center', fontsize = 8)
	plt.text(5, 65, 'sd =' + str(round(sdDesigned,2)), horizontalAlignment = 'center', fontsize = 8)
	plt.xlabel('Award Amount (Millions of $)', fontsize = 8)
	plt.title('Designed Programs', fontsize = 8)

	plt.savefig('awardAmountHistogram.png', dpi=300)
	plt.clf()

	# return some summary statistics 
	return(summaryStats)

def awardSizeHistogramPoster(df):
	data = np.zeros(len(df.awardAmount))
	for x in range(len(df.awardAmount)):
		a = str(df.awardAmount[x])
		b = a.replace('$','')
		c = b.replace(',','')
		data[x] = c
	data = data/1000000
	openData = df.OPEN*data
	openData = openData[openData != 0] 
	openData = np.append(openData, 0)
	openData = np.append(openData, 0)
	programData = abs(df.OPEN-1)*data 
	programData = programData[programData != 0]

	meanOpen = np.mean(openData)
	meanDesigned = np.mean(programData)
	sdOpen = np.std(openData)
	sdDesigned = np.std(programData)
	summaryStats = pd.DataFrame({'meanOpen':meanOpen, 'sdOpen':sdOpen, 'meanDesigned':meanDesigned, 'sdDesigned':sdDesigned}, index = [0])

	plt.figure(figsize=(8.5,5.5))
	plt.rcParams.update({'font.size': 18})
	plt.subplot(1,2,1)
	plt.hist(openData, bins=np.linspace(0, 10, 20))
	plt.ylim(0,80)
	plt.text(5, 75, 'n =' + str(len(openData)), horizontalAlignment='center')
	plt.text(5, 70, 'mean = ' + str(round(meanOpen, 2)), horizontalAlignment = 'center')
	plt.text(5, 65, 'sd =' + str(round(sdOpen,2)), horizontalAlignment = 'center')
	plt.ylabel('Number of Awards')
	#plt.xlabel('Award Amount (Millions of $)')
	plt.text(12, -10, 'Award Amount (Millions of $)', fontsize = 18, horizontalAlignment = 'center')
	plt.title('OPEN Projects')

	plt.subplot(1,2,2)
	plt.hist(programData[programData !=0], bins=np.linspace(0, 10, 20))
	plt.ylim(0,80)
	plt.text(5, 75, 'n =' + str(len(programData)), horizontalAlignment='center')
	plt.text(5, 70, 'mean = ' + str(round(meanDesigned, 2)), horizontalAlignment = 'center')
	plt.text(5, 65, 'sd =' + str(round(sdDesigned,2)), horizontalAlignment = 'center')
	#plt.xlabel('Award Amount (Millions of $)')

	plt.title('Designed Programs')

	plt.savefig('awardAmountHistogramPoster.png', dpi=300)
	plt.clf()

	# return some summary statistics 
	return(summaryStats)

def startDateHistogram(df):
	data = np.zeros(len(df.startDate))
	for x in range(len(df.startDate)):
		data[x] = df.startDate[x].year
	openData = df.OPEN*data
	openData = openData[openData != 0] 
	#openData = np.append(openData, 0)
	#openData = np.append(openData, 0)
	programData = abs(df.OPEN-1)*data 
	programData = programData[programData != 0]

	plt.figure(figsize=(7.5,4))
	plt.subplot(1,3,1)
	plt.hist(data, bins=[2009,2012, 2015, 2017])
	plt.ylim(0,250)

	plt.subplot(1,3,2)
	plt.hist(openData, bins = [2009,2012, 2015, 2017]) #bins = np.linspace(2010, 2018, 3))
	plt.ylim(0,250)

	plt.subplot(1,3,3)
	plt.hist(programData[programData !=0], bins = [2009, 2012, 2015, 2017]) #np.linspace(2010, 2018, 3))
	plt.ylim(0,250)

	plt.savefig('startDateHistogram.png', dpi=300)
	plt.clf()

def cleanCoderData(df):
	# remove active projects
	df = df[df.Status != 'nan']
	df = df[df.Status != 'ACTIVE']
	df = df[df.Status != 'EXCLUDE']
	df = df.reset_index(drop=True)
	return(df)

def matchCodedData(df, cf, col, notes):
	for n in range(cf.shape[0]):
		x = df.index[df['tagline']==cf.Tagline[n]].tolist()
		y = df.index[df['companies']==cf.Companies[n]].tolist()
		z = set(x).intersection(y)

		if len(z)<1:
			print(cf.Tagline[n], cf.Companies[n])
		if hasattr(cf, 'Outcome'):
			df[col][z] = cf.Outcome[n]
		elif hasattr(cf, 'Final Decision'):
			df[col][z] = cf['Final Decision'][n]
		if col == 'coder1':
			df['recipientType'][z] = cf['Recipient Type'][n]
			df['coder1notes'][z] = cf['Comments'][n]
		else:
			df[notes][z]=cf['Comments'][n]
	df = df.reset_index(drop=True)
	return(df)

def addCodedData(df):
	#x = pd.read_csv('coder1.csv')
	x = pd.read_csv('Coder1 - Sheet1.csv')
	y = cleanCoderData(x)
	df['recipientType'] = ['blank']*(df.shape[0])
	df['coder1'] = ['blank']*(df.shape[0])
	df['coder2'] = ['blank']*(df.shape[0])
	df['coder3'] = ['blank']*(df.shape[0])
	df['coder4'] = ['blank']*(df.shape[0])
	df['coder5'] = ['blank']*(df.shape[0])
	df['coder1notes'] = ['blank']*(df.shape[0])
	df['coder2notes'] = ['blank']*(df.shape[0])
	df['coder3notes'] = ['blank']*(df.shape[0])
	df['coder4notes'] = ['blank']*(df.shape[0])
	df['coder5notes'] = ['blank']*(df.shape[0])
	df['dummyCoder'] = ['blank']*(df.shape[0])
	
	df = matchCodedData(df, y, 'coder1', 'coder1notes')
	
	#x = pd.read_csv('dummyCoder.csv')
	#y = cleanCoderData(x)
	#df = matchCodedData(df, y, 'dummyCoder')

	x = pd.read_csv('Coder2.csv') #tom 
	y = cleanCoderData(x)
	df = matchCodedData(df, y, 'coder2', 'coder2notes')

	x = pd.read_csv('Resource Efficiency, Building Efficiency, Grid Projects - Projects.csv') #erin  
	y = cleanCoderData(x)
	df = matchCodedData(df, y, 'coder3', 'coder3notes')

	x = pd.read_csv('Transportation Fuels.csv') #sarah 
	y = cleanCoderData(x)
	df = matchCodedData(df, y, 'coder4', 'coder4notes')

	#x = pd.read_csv('coder5.csv') # jeff
	#y = cleanCoderData(x)
	#df = matchCodedData(df, y, 'coder5')


	return(df)

def cohensKappa(df, coderA, coderB):
	p0 = 0
	# keep only non-blank for coder B
	dropList = np.array([])
	for n in range(len(df[coderA])):
		if df[coderB][n] == 'blank':
			dropList = np.append(dropList, n)
	df = df.drop(dropList, axis = 0)
	df = df.reset_index(drop = True)
	for n in range(len(df[coderA])):
		if df[coderA][n] == df[coderB][n]:
			p0 = p0 + 1
	x = df[coderA].value_counts(sort=False)
	x1 = df[coderA].value_counts(sort=False).index.tolist()
	y = df[coderB].value_counts(sort=False)
	y1 = df[coderB].value_counts(sort=False).index.tolist()

	data = [[coderA, len(df[coderA])], [coderB, len(df[coderB])]]
	ck = pd.DataFrame(data, columns = ['Coder', 'Total'])
	ck['Persist'] = [x[x1.index('Persist')], y[y1.index('Persist')]]
	ck['Pivot'] = [x[x1.index('Pivot')], y[y1.index('Pivot')]]
	ck['Perish'] = [x[x1.index('Perish')], y[y1.index('Perish')]]

	pe = (np.product(ck.Persist/ck.Total)) + (np.product(ck.Pivot/ck.Total)) + (np.product(ck.Perish/ck.Total))
	c = (p0 - pe)/(1-pe)
	return(c)

def idDisagreements(df, coderA, coderB):
	dropList = np.array([])
	for n in range(len(df[coderA])):
		if df[coderB][n] == 'blank' or df[coderA][n] == df[coderB][n]:
			dropList = np.append(dropList, n)
	df = df.drop(dropList, axis = 0)
	df = df.reset_index(drop = True)
	outDF = df[['companies', 'description', 'endDate', 'program', 'projectStatus', 'projecturl', 'startDate', 'state', 'tagline', 'techCat1', 'techCat2', 'awardAmount', 'OPEN', 'recipientType', coderA, coderB, (coderA+'notes'), (coderB+'notes')]]
	
	# save a csv file 
	outDF.to_csv(('disagreements'+coderA+coderB+'.csv'))
	return(outDF)
		
def addFollowOnData(df):
	print('hello world')

def loadFinalData(df):
	x = pd.read_csv('Coder1 - Sheet1.csv') # change file name after this is done 
	y = cleanCoderData(x)
	df['recipientType'] = ['blank']*(df.shape[0])
	df['coder1'] = ['blank']*(df.shape[0])
	df['coder1notes'] = ['blank']*(df.shape[0])
	df['FinalDecision'] = ['blank']*(df.shape[0])
	df['Notes'] = ['blank']*(df.shape[0])
	df = matchCodedData(df, y, 'coder1', 'coder1notes')
	
	# add follow-on funding info 

	# add final decision info from other docs
	#x = pd.read_csv('disagreementscoder1coder2.csv') #tom 
	#y = cleanCoderData(x)
	#df = matchCodedData(df, y, 'FinalDecision', 'Notes')

	x = pd.read_csv('reconciledcoder1coder3.csv') # erin
	#y = cleanCoderData(x)
	df = matchCodedData(df, x, 'FinalDecision', 'Notes')

	x = pd.read_csv('reconciledcoder1coder4.csv') # sarah
	#y = cleanCoderData(x)
	df = matchCodedData(df, x, 'FinalDecision', 'Notes')

	#x = pd.read_csv('reconciledcoder1coder5.csv') # jeff
	#y = cleanCoderData(x)
	#df = matchCodedData(df, y, 'FinalDecision', 'Notes')

	# for now, keep only coder 1 data for blank info 
	for n in range(df.shape[0]):
		if df.FinalDecision[n] == 'blank':
			df.FinalDecision[n] = df.coder1[n]
	return(df)

def disagreementsSummary(df):
	print('hello world')
	#run some summary statistics to find commonalities in disagreements. look within coder pairs and as a group 

def chi2calc(df, outcomeCol):
	count_series = df.groupby(['OPEN', outcomeCol]).size()
	new_df = count_series.to_frame(name = 'breakdown').reset_index()
	new_df = new_df[new_df.FinalDecision !='blank']
	print(new_df)
	stat, p, dof, expected = stats.chi2_contingency([new_df.breakdown[0:3], new_df.breakdown[3:6]])# ddof = 2)
	return(stat, p, dof, expected)


