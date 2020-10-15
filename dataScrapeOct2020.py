from bs4 import BeautifulSoup
import pandas as pd
import requests
from lxml import html

def scrapeSummaryData(page_link):
	page_response = requests.get(page_link)
	page_content = BeautifulSoup(page_response.content, "html.parser")
	a = page_content.find_all(class_='view view-program-projects view-id-program_projects view-display-id-block_1 js-view-dom-id-9078bbf90e90f3311c8bb0dd3cf3c1e3f2fdfcde38918c57a0f317f5461016f5')
	#print(type(a), a)
	h = page_content.find_all('a', href = True)
	#print(len(h), type(h))
	holder = []
	for x in h:
		a = str(x)
		if a.find('technologies/projects/') > 0:
			b = a[9:]
			#b.find('hreflang')
			c = b[:(b.find('hreflang')-2)]
			#print(b)
			holder.append(c)
	#print(len(holder), type(holder), holder)
	h2 = list(set(holder))
	print(len(h2), type(h2), h2)
	
	l2 = 'https://arpa-e.energy.gov' + h2[0]
	print(l2)

	page_response = requests.get(l2)
	page_content = BeautifulSoup(page_response.content, "html.parser")
	b = page_content.find_all(class_ ='col-md-6 col-sm-12')
	print(len(b), b)

	#print(len(holder))
	#print(str(holder))
	#for x in holder:
	#	a = str(x)
	#	print(a)
		#b = a[138:]
		#c = b.strip( '</div>' )
		#companies.append(c)

	#dataOut = pd.DataFrame({'companies':companies, 'tagline':taglines, 
	#	'program':program, 'startDate':project_term1, 'endDate':project_term2, 
	#	'projectStatus':project_status, 'state':project_state,'description':project_description, 
	#	'techCat1':techCategories1, 'techCat2':techCategories2, 'projecturl':projecturl})
	#return(dataOut)

page_link = 'https://arpa-e.energy.gov/technologies/programs/adept' #'https://arpa-e.energy.gov/?q=project-listing&field_program_tid=All&field_project_state_value=All&field_project_status_value=2&term_node_tid_depth=All&sort_by=field_organization_value&sort_order=ASC'
y = scrapeSummaryData(page_link)