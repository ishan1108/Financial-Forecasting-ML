from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import *
from sklearn import cross_validation
from sklearn.metrics import *
import math
import timeit
import pickle
import random


classifier_rf = pickle.load(open('randomforest.model', 'rb'))

# The features we selected from the dataset
all_features =  ['DE Ratio','Trailing P/E','Price/Sales','Price/Book',
                 'Profit Margin','Operating Margin','Return on Assets',
                 'Return on Equity','Revenue Per Share','Market Cap',
                 'Enterprise Value','Forward P/E','PEG Ratio','Enterprise Value/Revenue',
                 'Enterprise Value/EBITDA','Revenue','Gross Profit',
                 'EBITDA','Net Income Avl to Common ','Diluted EPS',
                 'Earnings Growth','Revenue Growth','Total Cash','Total Cash Per Share',
                 'Total Debt','Current Ratio','Book Value Per Share','Cash Flow',
                 'Beta','Held by Insiders','Held by Institutions','Shares Short (as of',
                 'Short Ratio','Short % of Float','Shares Short (prior ']


# Loading the data from the CSV file
data_df = pd.read_csv("dataset.csv")

# Selecting the first 2000 set from the data
data_df = data_df[2000:]
data_df = data_df.reindex(np.random.permutation(data_df.index))

# Replacing the missing values
data_df = data_df.replace("NaN",0).replace("N/A",0)

X = np.array(data_df[all_features].values)
X_norm = X

# Changing the underperform with 0 and outperform with 1 to get numerical values
y = (data_df["Status"]
     .replace("underperform",0)
     .replace("outperform",1)
     .values.tolist())

ret = RandomForestRegressor(n_estimators=1001,max_depth=50, random_state=0)
ret.fit(X, y)

ret.predict(X)
print "Random Forest Regressor score     " + str(ret.score(X,y))