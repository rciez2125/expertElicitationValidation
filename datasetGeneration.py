# pulls projects from the ARPA-E website to create blank spreadsheets for project outcome categorization

from bs4 import BeautifulSoup
import pandas as pd
import requests
from scrips import scrapeSummaryData, scrapeFundingData
import csv 

# try not to run this every time, load data from csv file 

# Active projects
page_link = 'https://arpa-e.energy.gov/?q=project-listing&field_program_tid=All&field_project_state_value=All&field_project_status_value=1&term_node_tid_depth=All&sort_by=field_organization_value&sort_order=ASC'
x = scrapeSummaryData(page_link)

# Alumni projects
page_link = 'https://arpa-e.energy.gov/?q=project-listing&field_program_tid=All&field_project_state_value=All&field_project_status_value=2&term_node_tid_depth=All&sort_by=field_organization_value&sort_order=ASC'
y = scrapeSummaryData(page_link)

# Cancelled projects 
page_link = 'https://arpa-e.energy.gov/?q=project-listing&field_program_tid=All&field_project_state_value=All&field_project_status_value=3&term_node_tid_depth=All&sort_by=field_organization_value&sort_order=ASC'
z = scrapeSummaryData(page_link)

result = pd.concat([x, y, z], axis=0)
resultOut = result.reset_index(drop = True)
#resultOut.to_csv('arpaeSummaryData.csv')
#resultOut.to_csv('arpaeSummaryDataJune2020.csv')

# this takes ~15-20 minutes to run
#a = scrapeFundingData('arpaeSummaryData.csv')
#a.to_csv('arpaeSummaryDataWithAward.csv')