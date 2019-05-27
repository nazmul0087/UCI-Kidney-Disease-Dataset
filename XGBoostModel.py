#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 00:39:31 2019

@author: Tony Hunag
"""

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt
import seaborn as sns
    
#one-hot file loading
def load(filepath):
    raw_data = pd.read_csv(filepath)
    #Create a data frame for column storage
    processed_columns = pd.DataFrame({})
    for col in raw_data:
        #col_datatype = raw_data[col].dtype
        #Check the column for dtype object or unique value < 7
        if raw_data[col].dtype == 'object' or raw_data[col].nunique() < 7:
            df = pd.get_dummies(raw_data[col], prefix=col)
            processed_columns = pd.concat([processed_columns, df], axis=1)
        else:
            processed_columns = pd.concat([processed_columns, raw_data[col]], axis=1)
    return processed_columns



X = load("X.csv")
df2 = pd.read_csv("final.csv")
Y = df2.loc[:,['class']]


#seed = 7
test_size = 0.25

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=False)

feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)

#neighbortests = [3,5,7,9,11]
maxDepth = [2,3,4,5,6,7,8,9,10,11,12]
accDepth =[]


for x in maxDepth:
    model = XGBClassifier(max_depth=x, learning_rate=0.1,
                                n_estimators=100,
                                silent=True,   objective='binary:hinge',
                                nthread=-1, gamma=0,
                                min_child_weight=2, max_delta_step=0, subsample=0.8,
                                colsample_bytree=0.6,
                                base_score=0.5,
                                seed=0, missing=None)
    all_accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)
    meanAverage = np.mean(all_accuracies)
    print('max depth count = {}. accuracy = {}'.format(x,meanAverage))
    accDepth.append(meanAverage)
    


#Creating the histogram
plt.plot(maxDepth, accDepth)
plt.title("XGBoost Model")
plt.ylabel("Accuracy")
plt.xlabel("Max Depth of Tree")
plt.show()

correlation_matrix = df2.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix,xticklabels=True, yticklabels=True, annot=True)
plt.show()
