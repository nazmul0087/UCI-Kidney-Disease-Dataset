#!/usr/bin/env python3

import numpy as np
import pandas as pd

df = pd.read_csv("chronic_kidney_disease_full.csv")
df2 = df.copy()

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