from bs4 import BeautifulSoup
import requests
import time 
import pandas as pd
import numpy as np
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
