import pandas as pd 
import numpy as np 
#from datetime import datetime
#from dateutil.parser import parse
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize 
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
import random

size_Open = 110 
size_Designed = 350 

# create some fake data that approximates a 95% success rate for open projects and a 10% success rate for designed
p_open = [0.38, 0.41, 0.21] #perish, pivot, persist
p_open = [0.21, 0.38, 0.41]
p_designed = [0.35, 0.46, 0.19] #perish, pivot, persist
p_designed = [0.19, 0.35, 0.46]

#p_open = [0.41, 0.38, 0.21]
#p_designed = [0.46, 0.35, 0.19] #pivot, perish, persist
# check that both of them = 1
if sum(p_open)!= 1 or sum(p_designed)!=1:
    print('test probabilities not equal to 100%')

x_open = np.random.multinomial(size_Open, p_open)
#print(np.unique(x_open))
x_designed = np.random.multinomial(size_Designed, p_designed)
#print(np.unique(x_designed))

x_open = [51, 65, 29] # perish, pivot, persist
x_designed = [81, 104, 37] # perish, pivot, persist
size_Open = sum(x_open)
size_Designed = sum(x_designed)

stat, p, dof, expected = stats.chi2_contingency([x_designed, x_open])# ddof = 2)
print('stat', stat, 'p', p, 'expected', expected)

def generateData(x_open, x_designed):
    test_Open = np.vstack((np.ones(size_Open), np.hstack((np.ones(x_open[0]), 2*np.ones(x_open[1]), 3*np.ones(x_open[2])))))
    test_Designed = np.vstack((np.zeros(size_Designed), np.hstack((np.ones(x_designed[0]), 2*np.ones(x_designed[1]), 3*np.ones(x_designed[2])))))
    testData = np.transpose(np.hstack((test_Open, test_Designed)))
    testData = pd.DataFrame(data = testData, columns = ['OpenDummy', 'predicted'])

    print(testData.groupby('predicted').count())
    print(testData.groupby('OpenDummy').count())

    # add some fake funding, start date year data
    testData['awardAmount'] = np.random.randint(100000, high = 5000000, size = (size_Open+size_Designed)) 
    testData['awardeeType'] = np.random.randint(0, 2, size = (size_Open+size_Designed))

    #make these more realistic based on actual probabilities 
    testData['techCategory'] = np.random.randint(0, 12, size = (size_Open+size_Designed))
    testData['partners'] = np.random.randint(0, 2, size = (size_Open+size_Designed))

    # models work with scaled from 0 data, but not real years 
    testData['startingYear'] = np.random.randint(2009, high = 2017, size = (size_Open+size_Designed)) 
    testData['early'] = np.zeros((size_Open+size_Designed))
    testData['middle'] = np.zeros((size_Open+size_Designed))
    testData['latest'] = np.zeros((size_Open+size_Designed))
    for n in range(size_Open+size_Designed):
        if testData.startingYear[n]<2013:
            testData.set_value([n], 'early', 1)
        elif testData.startingYear[n] < 2016:
            testData.set_value([n], 'middle', 1)
        else:
            testData.set_value([n], 'latest', 1)
    # try some different distributions for this 
    testData['followOnFunds'] = testData['predicted'] * np.random.randint(0, high = 4000000, size = (size_Open+size_Designed))
    return testData
testData = generateData(x_open, x_designed)

# solve the basic mnl function
exog = sm.add_constant(testData.OpenDummy)
mdl0 = sm.MNLogit(testData.predicted, exog)
mdl0_fit = mdl0.fit()
#print(mdl0_fit.summary())
#print(x_open)
#print(x_designed)
#print(mdl0_fit.params)

# solve the basic mnl function with starting year fixed effects
def runBasicModel(target):
    #exog = sm.add_constant(testData[['OpenDummy', 'startingYear']])
    #exog = testData[['OpenDummy', 'startingYear']]
    exog = testData.OpenDummy
    mdl1 = sm.MNLogit(target, exog).fit()
    print(mdl1.summary())
    return(mdl1.params, mdl1.pvalues, mdl1._results.conf_int())
runBasicModel(testData.predicted)

# solve the mnl function open + funding
def runAwardAmountModel(target):
    exog = sm.add_constant(testData[['OpenDummy','early', 'middle', 'latest']])
    mdl1 = sm.MNLogit(target, exog)
    mdl1_fit = mdl1.fit()
    print(mdl1_fit.summary())

def runAwardeeTypeModel(target):
    #solve the mnl function open + awardee type
    exog = testData[['OpenDummy', 'awardeeType', 'startingYear']]
    #adding a constant did basically nothing, do we have to include? why/why not? 
    exog = sm.add_constant(testData[['OpenDummy', 'awardeeType', 'startingYear']])
    mdl3 = sm.MNLogit(target, exog)
    mdl3_fit = mdl3.fit()
    print(mdl3_fit.summary())

    #exog = testData[['OpenDummy', 'awardeeType', 'early', 'middle', 'latest']]
    #mdl3 = sm.MNLogit(testData.predicted, exog)
    #mdl3_fit = mdl3.fit()
    #print(mdl3_fit.summary())

# solve the mnl function open + tech category
def runTechCatModel():
    exog = testData[['OpenDummy', 'techCategory', 'startingYear']]
    mdl4 = sm.MNLogit(target, exog)
    mdl4_fit = mdl4.fit()
    print(mdl4_fit.summary())

    exog = testData[['OpenDummy', 'techCategory', 'early', 'middle', 'latest']]
    mdl4 = sm.MNLogit(target, exog)
    mdl4_fit = mdl4.fit()
    print(mdl4_fit.summary())

# solve the mnl function open + partners
def runPartnersModel(target):
    exog = testData[['OpenDummy', 'partners', 'startingYear']]
    mdl5 = sm.MNLogit(target, exog)
    #mdl5_fit = mdl5.fit()
    #print(mdl5_fit.summary())

    #exog = testData[['OpenDummy', 'partners', 'early', 'middle', 'latest']]
    #mdl5 = sm.MNLogit(testData.predicted, exog)
    #mdl5_fit = mdl5.fit()
    #print(mdl5_fit.summary())

# solve the mnl function with all variables considered
def runAllVarsModel(target):
    exog = testData[['OpenDummy', 'awardAmount', 'awardeeType', 'techCategory', 'partners', 'startingYear']]
    mdl6 = sm.MNLogit(target, exog)
    mdl6_fit = mdl6.fit()
    print(mdl6_fit.summary())

    exog = testData[['OpenDummy', 'awardAmount', 'awardeeType', 'techCategory', 'partners', 'early', 'middle', 'latest']]
    mdl6 = sm.MNLogit(target, exog)
    mdl6_fit = mdl6.fit()
    print(mdl6_fit.summary())

# run a linear regression on the follow-on funding amount 
def linModBasic():
    #exog = sm.add_constant(testData[['OpenDummy', 'startingYear']])
    #exog = testData[['OpenDummy', 'startingYear']]
    exog = testData.OpenDummy
    model = sm.OLS(testData.followOnFunds, exog).fit()
    predictions = model.predict(exog)
    print(model.summary())
    print(model.conf_int(alpha=0.05, cols = None)) #extracting the confidence intervals works for ols
#linModBasic()

def linModAwardAmt():
    exog = sm.add_constant(testData[['OpenDummy', 'awardAmount', 'startingYear']])
    model = sm.OLS(testData.followOnFunds, exog).fit()
    print(model.summary())
#linModAwardAmt()

def runSimulation(ow, dw):
    p_open = np.zeros(3)
    p_designed = np.zeros(3)
    for n in range(100):
        x = np.random.rand(1)
        if x<ow[0]:
            p_open[0] = p_open[0] + 1
        elif x<sum(ow[0:2]):
            p_open[1] = p_open[1] + 1
        else:
            p_open[2] = p_open[2] + 1
    for n in range(100):
        x = np.random.rand(1)
        if x<dw[0]:
            p_designed[0] = p_designed[0] + 1
        elif x<sum(dw[0:2]):
            p_designed[1] = p_designed[1] + 1
        else:
            p_designed[2] = p_designed[2] + 1
    
    #p_open[1] = p_open[1]-0.05 # test to see if this matters
    p_open = p_open/sum(p_open) #perish, pivot, persist
    #p_designed = np.random.rand(3)
    p_designed = p_designed/sum(p_designed) #perish, pivot, persist
    #print(p_designed)

    x_open = np.random.multinomial(size_Open, p_open)
    x_designed = np.random.multinomial(size_Designed, p_designed)
    stat, p, dof, expected = stats.chi2_contingency([x_designed, x_open])# ddof = 2)
    testData = generateData(x_open, x_designed)
    coeffs, pvals, ci = runBasicModel(testData.predicted)
    outData = {'chisquarep': p, 'coeffs': coeffs, 'pvals': pvals, 'ci':ci, 'p_designed':p_designed} # any other data to be returned?
    return outData

#numSims = 100
#coeffsHolder = np.zeros((numSims,2))
#lbHolder = np.zeros((numSims,2))
#ubHolder = np.zeros((numSims,2))
#pvalHolder = np.zeros((numSims,2))
#designedHolder = np.zeros((numSims,3))
#chiSquares = np.zeros(numSims)
#for n in range(numSims):
 #   x = runSimulation(p_open, p_designed)#[0.7, 0.2, 0.1], [0.6, 0.25, 0.15])
  #  coeffsHolder[n,:] = x['coeffs'].iloc[0]
   # lbHolder[n,0] = x['ci'][0,0,0]
    #lbHolder[n,1] = x['ci'][1,0,0]
#    ubHolder[n,0] = x['ci'][0,0,1]
 #   ubHolder[n,1] = x['ci'][1,0,1]
  #  pvalHolder[n,:] = x['pvals'].iloc[0]
   # chiSquares[n] = x['chisquarep']
    #designedHolder[n,:] = x['p_designed']

def makePlots():
    # make a plot
    plt.figure(figsize=(4,4))
    diffTotal = np.zeros(3)
    perfTotal = 0
    plt.plot(np.linspace(1,numSims,numSims), coeffsHolder[:,0], '.b')
    plt.plot(np.linspace(1,numSims,numSims), coeffsHolder[:,1], '.r')
    for n in range(numSims):
        plt.plot([n+1,n+1], [lbHolder[n,0], ubHolder[n,0]], '-b')
        plt.plot([n+1,n+1], [lbHolder[n,1], ubHolder[n,1]], '-r')
        if lbHolder[n,0]>ubHolder[n,1] or lbHolder[n,1]>ubHolder[n,0]: # different from each other
            if lbHolder[n,0]<0 and ubHolder[n,0]> 0: #different from zero
                diffTotal[0] = diffTotal[0]+0
            else: diffTotal[0] = diffTotal[0] + 1
            if lbHolder[n,1]<0 and ubHolder[n,1]>0: #different from zero
                diffTotal[1] = diffTotal[1]+0
            else: diffTotal[1] = diffTotal[1] + 1
            if ((lbHolder[n,0]<0 and ubHolder[n,0]<0) or (lbHolder[n,0]>0 and ubHolder[n,0]>0)) and ((lbHolder[n,1]<0 and ubHolder[n,1]<0) or (lbHolder[n,1]>0 and ubHolder[n,1]>0)):
                diffTotal[2] = diffTotal[2] + 1 
            if pvalHolder[n,0] <0.05 and pvalHolder[n,1] <0.05 and chiSquares[n]<0.05:
                perfTotal = perfTotal + 1
    plt.ylim(-10,10)
    plt.plot([0,100], [0, 0], '-k')
    plt.xlim(0,numSims+1)
    plt.xlabel('Simulations')
    plt.ylabel('Coefficients on OPEN Program Dummy Var')
    plt.legend(('Pivot', 'Persist'))
    print('different total', diffTotal/numSims)
    print('perfect total', perfTotal/numSims)
    plt.savefig('coefficients.png')

    plt.clf()

    plt.figure(figsize=(4,4))
    plt.plot(np.linspace(1,numSims,numSims), pvalHolder[:,0], '.b')
    plt.plot(np.linspace(1,numSims,numSims), pvalHolder[:,1], '.r')
    plt.plot(np.linspace(1,numSims,numSims), chiSquares, '.', color = [0.5, 0.5, 0.5])
    plt.plot([0,100],[0.05, 0.05], '-k')
    plt.plot([0,100],[0.1, 0.1], '--k')
    plt.xlabel('Simulations')
    plt.ylabel('p-value on OPEN Program Dummy Var')
    plt.legend(('Pivot', 'Persist', 'Chi Square'))
    plt.xlim(0,numSims+1)
    plt.savefig('pvals.png')

    print('Chisquares',(chiSquares<0.05).sum())
    print('Chiquares', (chiSquares<0.1).sum())


    plt.figure(figsize=(4,4))
    plt.plot(np.repeat(1, numSims), designedHolder[:,0], '.b', alpha = 0.4)
    plt.plot(np.repeat(2, numSims), designedHolder[:,1], '.g', alpha = 0.4)
    plt.plot(np.repeat(3, numSims), designedHolder[:,2], '.r', alpha = 0.4)
    plt.ylabel('probability of project outcome')
    plt.savefig('designedProbs.png')
    print(np.median(designedHolder[:,0]))
    print(np.median(designedHolder[:,1]))
    print(np.median(designedHolder[:,2]))
#makePlots()