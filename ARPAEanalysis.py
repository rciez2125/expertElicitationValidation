# master file that runs each of the components for dataset creation/formatting, 
# pretesting + power analysis, figure generation, and final analysis
from scrips import addCodedData, cohensKappa
from datasetAnalysis import runDatasetAnalysis
# scrape data from the website
# comment this out to save time (pulling funding info takes ~15-20 minutes)
# import datasetGeneration

# run preliminary dataset analysis
Z = runDatasetAnalysis()
# run the power analysis

#import powerAnalysis

# run pretesting

# compile coded data
df = addCodedData(Z)

# run cohen's kappa calculations
kDummy = cohensKappa(df, 'coder1', 'dummyCoder')
print(kDummy)
# complie follow-on funding data

# run final analysis 