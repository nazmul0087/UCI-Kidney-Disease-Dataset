#!/usr/bin/env python3

import numpy as np
import pandas as pd

#Splitting the 2 data types to categorical and numerical

#Numerical
df = pd.read_csv("chronic_kidney_disease_full.csv")
#Categorical
df2 = df.copy()

#Categorical data

#Red blood cells replace all '?' values with the majority value 'normal'
rbcNormal = df2['rbc'].str.replace('?', 'normal')
df2['rbc'] = rbcNormal

#Pus cells replace all '?' values with the majority value 'normal'
pcNormal = df2['pc'].str.replace('?', 'normal')
df2['pc'] = pcNormal

#Pus cell clumps replace all '?' values with the majority value 'notpresent'
pccNotpresent = df2['pcc'].str.replace('?', 'notpresent')
df2['pcc'] = pccNotpresent

#Bacteria replace all '?' values with the majority value 'notpresent'
BaNotpresent = df2['ba'].str.replace('?', 'notpresent')
df2['ba'] = BaNotpresent

#appetitie replace all '?' values with the majority value 'good'
appetGood = df2['appet'].str.replace('?', 'good')
df2['appet'] = appetGood

#Pedal edema replace all '?' values with the majority value 'no'
peNo = df2['pe'].str.replace('?', 'no')
df2['pe'] = peNo

#Anemia edema replace all '?' values with the majority value 'no'
aneNo = df2['ane'].str.replace('?', 'no')
df2['ane'] = aneNo

#Make class values into one hot. ckd = 0. nonckd = 1
classckd = df['class'].str.replace('ckd', '0')
df['class'] = classckd

classnotckd = df['class'].str.replace('not0', '1')
df['class'] = classnotckd

#Drop the categorical columns from the first dataframe
df.drop(['rbc'],1, inplace=True)
df.drop(['pc'],1, inplace=True)
df.drop(['pcc'],1, inplace=True)
df.drop(['ba'],1, inplace=True)
df.drop(['htn'],1, inplace=True)
df.drop(['dm'],1, inplace=True)
df.drop(['cad'],1, inplace=True)
df.drop(['appet'],1, inplace=True)
df.drop(['pe'],1, inplace=True)
df.drop(['ane'],1, inplace=True)
df.drop(['su'],1, inplace=True)
df.drop(['sg'],1, inplace=True)
df.drop(['al'],1, inplace=True)

#Contains all the numerical columns 
X = df.drop(['class'], axis=1, inplace=False)
Y = df.loc[:,['class']]
X.replace('?', np.nan, inplace=True)

columnNames = list(X.columns.values)

#Drop the numerical columns from df2
for names in columnNames:
    df2.drop([names],1, inplace=True)

df2.drop(['class'], axis=1, inplace=True)


#Fill out the numerical columns with the mean value.
for names in columnNames:
    testcolumn = np.array(X[names]).astype(float)
    mean = np.nanmean(testcolumn.astype(float))
    testcolumn[np.isnan(testcolumn)]=mean
    X[names] = testcolumn
    
#Combine the numerical and categorical data together
df_col = pd.concat([df2,X], axis=1)

X = df_col

#Create X for one hot loading
X.to_csv("X.csv",index=False)

#Create the excel first. Cleaned csv dataa
finalcsv = pd.concat([X,Y], axis=1)