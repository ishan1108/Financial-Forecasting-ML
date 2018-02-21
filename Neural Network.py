# Importing the necessary libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, tree
from sklearn import svm
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.externals import *
from sklearn import cross_validation
from sklearn.metrics import *
import math
import timeit
import pickle
import random


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
data_df = data_df[:2000]
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

# Pre-processing
X = preprocessing.scale(X)

# Selecting the 20 best features
X = SelectKBest(k=20).fit_transform(X, y)
samples = len(X)


def getSet(original, indeces):
    ret = []

    for ind in indeces:
        ret.append(original[ind])
    return ret


# To calculate the accuracy, precision, recall & f1 for the machine learning algorithms
def Analysis(neural_network):
    global X, y
    test_sample_size = samples / 6
    analysis_times = 1
    split = cross_validation.ShuffleSplit(samples, n_iter=analysis_times, test_size=0.05, random_state=0)
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_times = analysis_times


    for train, test in split:
        train_data = getSet(X, train)
        test_data = getSet(X, test)

        train_values = getSet(y, train)
        current_values = getSet(y, test)

        clf = neural_network[0]
        clf.fit(train_data, train_values)
        

        total_num = 0
        truepositive = 0
        truenegative = 0
        falsenegative = 0
        falsepositive = 0

        for j in range(len(test_data)):
            current_data = test_data[j]
            current_value = current_values[j]
            
            # predict the value of current
            predicted_value = clf.predict([current_data])[0]
            total_num += 1

            if predicted_value == current_value:
                if current_value == 1:
                    truepositive += 1
                    
                else:
                    truenegative += 1
            else:
                if current_value == 0:
                    falsepositive += 1
                    
                else:
                    falsenegative += 1
        
        if(truepositive != 0):
            accuracy = float(truepositive + truenegative) / total_num
            precision = float(truepositive) / (truepositive + falsepositive)
            recall = float(truepositive) / (truepositive + falsenegative)
            f1 = 2 * precision * recall / (precision + recall)
        else:
            0

        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        
    final_accuracy = total_accuracy / total_times
    final_precision = total_precision / total_times
    final_recall = total_recall / total_times
    final_f1 = total_f1 / total_times
    classifier_name = neural_network[1]
    
    print "Classifier is " + classifier_name 
    print "Accuracy = " + str(final_accuracy)
    print "Precision = " + str(final_precision)
    print "Recall = " + str(final_recall) 
    print "F1 = " + str(final_f1)
    
    return final_accuracy, final_precision, final_recall, final_f1

degree_ = len(all_features)
infos = []


# Neural Network function

ne= MLPClassifier(hidden_layer_sizes=4000,max_iter=1001)


# Saving the neural network model for future predictions
filename = "neural network.model"
joblib.dump(ne, filename)
infos.append([ne, "Neural Network"])


# Using the analysis function to get the values of
# accuracy, precision, recall and f1
for neural_network in infos:
     accuracy, precision, recall, f1 = Analysis(neural_network)

N = len(infos)
ind = np.arange(N)
width = 0.2 

# Plotting the bar chart of accuracy, precision, recall & f1

fig, ax = plt.subplots()

ax.set_ylabel('Rate')
ax.set_xticks(ind + width)
ax.set_xticklabels('Neural Network')

bar1 = ax.bar(ind, accuracy, width, color='r')
bar2 = ax.bar(ind + width, precision, width, color='y')
bar3 = ax.bar(ind + width * 2, recall, width, color='g')
bar4 = ax.bar(ind + width * 3, f1, width, color='b')



ax.legend((bar1[0], bar2[0], bar3[0], bar4[0]),
          ('ROC', 'Sensitivity', 'Specificity', 'f1'), loc=2,prop={'size':8})

# Saving the chart to a pdf file
plt.savefig('neural network_barChart.pdf')