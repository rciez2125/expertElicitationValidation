import pandas as pd 
import numpy as np 
from datetime import datetime
from dateutil.parser import parse

# load the CSV file with the data

x = pd.read_csv('arpaeSummaryData.csv')
y = x.drop(x.columns[0], axis=1)
a = [datetime.strptime(n, '%m/%d/%Y') for n in y.startDate]
b = [datetime.strptime(n, '%m/%d/%Y') for n in y.endDate]
y.startDate = a
y.endDate = b

# add a binary value for OPEN Projects
Z = []
for n in y.program:
    a = str(n)
    b = a.find('OPEN')
    Z.append(b)
y['OPEN'] = Z
y.OPEN = y.OPEN+1

# basic summary stats
a = 0
b = 0
c = 0
for n in range(len(y.projectStatus)):
    if y.projectStatus[n] == 'ACTIVE':
        a = a + y.OPEN[n]
    if y.projectStatus[n] == 'ALUMNI':
    	b = b + y.OPEN[n]
    if y.projectStatus[n] == 'CANCELLED':
        c = c + y.OPEN[n]

print('Active OPEN Projects')
print(a)
print('Alumni OPEN projects')
print(b)
print('Cancelled OPEN projects')
print(c)

#print(y.projectStatus['ACTIVE'].value_counts())
print('All active projects')
print(y.loc[y.projectStatus == 'ACTIVE', 'projectStatus'].count())

print('All alumni projects')
print(y.loc[y.projectStatus == 'ALUMNI', 'projectStatus'].count())

print('All cancelled projects')
print(y.loc[y.projectStatus == 'CANCELLED', 'projectStatus'].count())