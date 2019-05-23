#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt

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
neighbors = [1,3,5,7,9,11,13,15]
accDepth =[]


for x in neighbors:
    model = KNeighborsClassifier(n_neighbors = x)
    all_accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)
    meanAverage = np.mean(all_accuracies)
    print('Neighbor = {}. accuracy = {}'.format(x,meanAverage))
    accDepth.append(meanAverage)

#Creating the visualisation
plt.plot(neighbors, accDepth)
plt.title("K-NN Model")
plt.ylabel("Accuracy")
plt.xlabel("K-neighbours")
plt.show()
