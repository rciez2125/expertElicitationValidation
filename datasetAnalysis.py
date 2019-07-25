import pandas as pd 
import numpy as np 
from datetime import datetime
from dateutil.parser import parse
from scrips import awardSizeHistogram, awardSizeHistogramPoster, startDateHistogram
import matplotlib.pyplot as plt

# load the CSV file with the data

x = pd.read_csv('arpaeSummaryDataWithAward.csv')
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

# remove cancelled projects that would be active if they had finished
dropList = []
fixedDate = datetime(2016, 1, 1, hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
for n in range(len(y.projectStatus)):
    if y.projectStatus[n] == 'CANCELLED':
        if y.startDate[n] > fixedDate:
            dropList.append(n)
print(dropList)
z = y.drop(y.index[dropList])
z = z.reset_index(drop=True)
print(z.shape)

# basic summary stats, number of projects
a = 0
b = 0
c = 0
d = 0 
for n in range(len(z.projectStatus)):
    if z.projectStatus[n] == 'ACTIVE':
        a = a + z.OPEN[n]
    if z.projectStatus[n] == 'ALUMNI':
    	b = b + z.OPEN[n]
    if z.projectStatus[n] == 'CANCELLED':
        c = c + z.OPEN[n]
    if z.projectStatus[n] == 'RECENTLYCANCELLED':
        d = d + z.OPEN[n]

print('Active OPEN Projects')
print(a)
print('Alumni OPEN projects')
print(b)
print('Cancelled OPEN projects')
print(c)
print('Recently Cancelled OPEN projects')
print(d)
print('Total OPEN projects')
print(a+b+c+d)

#print(y.projectStatus['ACTIVE'].value_counts())
print('All active projects')
print(z.loc[z.projectStatus == 'ACTIVE', 'projectStatus'].count())

print('All alumni projects')
print(z.loc[z.projectStatus == 'ALUMNI', 'projectStatus'].count())

print('All cancelled projects')
print(z.loc[z.projectStatus == 'CANCELLED', 'projectStatus'].count())

print('All recently cancelled projects')
print(z.loc[z.projectStatus == 'RECENTLYCANCELLED', 'projectStatus'].count())

print('All projects')
print(z.shape)

# make a dataframe only of alumni and cancelled projects
dropList = []
for n in range(len(z.projectStatus)):
    if z.projectStatus[n] == 'ACTIVE':
        dropList.append(n)
    elif z.projectStatus[n] == 'RECENTLYCANCELLED':
        dropList.append(n)
Z = z.drop(z.index[dropList])
Z = Z.reset_index(drop=True)
print(Z.shape)

# make a histogram of award sizes
sumStats = awardSizeHistogram(Z)
print(sumStats)
#a = awardSizeHistogramPoster(Z)

# how many programs are there?
print('Number of programs')
print(len(Z.program.unique()))
print('Program Names')
print(Z.program.unique())

# make a histogram of projects by starting year
t = Z.startDate[0].year
print(type(t))
print(Z.startDate[0].year)
print(type(Z.startDate[0]))
startDateHistogram(Z)

print('Tech categories')
print(Z['techCat1'].value_counts())

print('Award Amount per tech category')
n = np.unique(Z.techCat1)
awardTotals = np.ones(len(n))
for m in range(len(n)):
    awardTotals[m] = np.round(Z.loc[Z['techCat1'] == n[m], 'awardAmount'].sum()/1000000, 2)
print(awardTotals)
print(n)
