import pandas as pd 
import numpy as np 
from datetime import datetime
from dateutil.parser import parse
from scrips import awardSizeHistogram, awardSizeHistogramPoster, startDateHistogram
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# load the CSV file with the data
def runDatasetAnalysis():
    x = pd.read_csv('Data/arpaeSummaryDataWithAward.csv')
    y = x.drop(x.columns[0], axis=1)
    a = [datetime.strptime(n, '%m/%d/%Y') for n in y.startDate]
    b = [datetime.strptime(n, '%m/%d/%Y') for n in y.endDate]
    y.startDate = a
    y.endDate = b

    # add a binary value for OPEN Projects
    print(y.shape[0])
    Z = np.zeros((y.shape[0],1))
    for n in range(len(y.program)):
        a = str(y.program[n])
        b = a.find('OPEN')
        if b != -1 or a == 'IDEAS':
            Z[n] = 1
    y['OPEN'] = Z

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

    # add the costTarget Data
    Z['costTarget'] = ['blank']*Z.shape[0]
    targetData = pd.read_csv("Data/costTargets.csv")
    for n in range(targetData.shape[0]):
        for m in range(Z.shape[0]):
            if targetData.Program[n] == Z.program[m]:
                Z.costTarget[m] = targetData.CostTarget[n]
    print(Z.head(10))

    # make a histogram of award sizes
    sumStats = awardSizeHistogram(Z)
    print(sumStats)
    #a = awardSizeHistogramPoster(Z)

    # how many programs are there?
    print('Number of programs')
    print(len(Z.program.unique()))
    print('Program Names')
    print(Z.program.unique())

    # how many projects have cost targets
    print('cost target counts')
    print(Z.costTarget.value_counts())

    print('cost target total funding')
    print(Z.groupby('costTarget')['awardAmount'].sum())

    print('OPEN total funding')
    print(Z.groupby('OPEN')['awardAmount'].sum())
   
    print('total awarded')
    print(Z['awardAmount'].sum())

    print('OPEN unspecified funding')
    print(Z.groupby('OPEN')['awardAmount'].sum()[1]/Z['awardAmount'].sum())

    print('other unspecified funding')
    print((Z.groupby('costTarget')['awardAmount'].sum()[0]-Z.groupby('OPEN')['awardAmount'].sum()[1])/Z['awardAmount'].sum())

    print('specified funding')
    print(Z.groupby('costTarget')['awardAmount'].sum()[1]/Z['awardAmount'].sum())

    # make a histogram of projects by starting year
    t = Z.startDate[0].year
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

    return(Z)