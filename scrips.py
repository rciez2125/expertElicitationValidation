from bs4 import BeautifulSoup
import requests
import pandas as pd

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
	for x in holder:
		a = str(x)
		b = a[146:]
		c = b.strip( '</div>' ) 
		d = c.strip( '</h2>' )   
		f = d.strip( '</a>' )
		h = f.rfind('>')+1
		k = f[h:]
		taglines.append(k)
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
		d = c.strip('</div>')
		f = d.strip('</p>\n')
		project_description.append(f)
	dataOut = pd.DataFrame({'companies':companies, 'tagline':taglines, 'program':program, 'startDate':project_term1, 'endDate':project_term2, 'projectStatus':project_status, 'state':project_state,'description':project_description})
	return(dataOut)