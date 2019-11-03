# master file that runs each of the components for dataset creation/formatting, 
# pretesting + power analysis, figure generation, and final analysis
from scrips import addCodedData, cohensKappa, idDisagreements, loadFinalData, disagreementsSummary
from datasetAnalysis import runDatasetAnalysis
# scrape data from the website
# comment this out to save time (pulling funding info takes ~15-20 minutes)
# import datasetGeneration

# run preliminary dataset analysis
Z = runDatasetAnalysis()

# run the power analysis
#import powerAnalysis

# compile coded data
df = addCodedData(Z)
#print(df)


# run cohen's kappa calculations
k14 = cohensKappa(df, 'coder1', 'coder4')
print('k14', k14)
d = idDisagreements(df, 'coder1', 'coder4')
x = disagreementsSummary(d)
print(x)
print(type(x))

k12 = cohensKappa(df, 'coder1', 'coder2')
print('k12', k12)
d = idDisagreements(df, 'coder1', 'coder2')
x = disagreementsSummary(d)
print(x)

k13 = cohensKappa(df, 'coder1', 'coder3')
print('k13', k13)
d = idDisagreements(df, 'coder1', 'coder3')
x = disagreementsSummary(d)
print(x)

# tom, jeff, erin, sarah 

# load the final data 
df = loadFinalData(Z)
df.to_csv(('FinalData.csv'))
df['endYr'] = ""
for n in range(df.shape[0]):
	df.endYr[n] = df.endDate[n].year
x = df.endYr.value_counts()
print(x)

# run final analysis 
