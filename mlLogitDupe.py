import numpy as np 
import pandas as pd 
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize 
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex
from matplotlib.colors import to_rgb
import random

def model1(beta):
	a1 = 