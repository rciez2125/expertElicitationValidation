import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from datetime import datetime

df = pd.read_csv('arpaeSummaryDataJune2020.csv')
print(df.columns)
df = df.drop(columns = 'Unnamed: 0')
print(df.columns)

df = df.sort_values(['companies']).reset_index(drop=True)
df.to_csv('sortedList.csv')

df['companyCopy'] = df.companies.str.strip()
#('sortedList.csv')
#companyCopy = x
duplicateRowsDF = df[df.duplicated(subset = 'companyCopy')]

#duplicateRowsDF = duplicateRowsDF.sort_values(['companies']).reset_index(drop = True)
duplicateRowsDF.to_csv('dupes.csv')