# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 18:35:16 2018

@author: MMartchevskii
"""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib import rcParams

# Set random seed
np.random.seed(0)

#################################################################
# Manual loading of training and test datasets
filedir = 'G:/TDMS_DATA/CCT4_Pico6/SORTED/MultiWaveR'

type0files = ['Feat_8_9_2017_1_56_57_PM_A03_Q1_9677A.csv']

type1files = ['Feat_8_24_2017_2_46_30_PM_A129_Q103_15986A.csv']

tefiles = ['Feat_8_24_2017_2_46_30_PM_A129_Q103_15986A.csv','Feat_8_18_2017_2_18_48_PM_A100_Q79_16177A.csv','Feat_8_17_2017_5_33_46_PM_A84_Q64_15613A.csv',
           'Feat_8_16_2017_9_10_29_AM_A67_Q49_15029A.csv','Feat_8_14_2017_4_38_37_PM_A50_Q35_14160A.csv', 
           'Feat_8_11_2017_9_25_27_AM_A35_Q25_13638A.csv', 'Feat_8_10_2017_11_57_53_AM_A15_Q11_12361A.csv',
           'Feat_8_10_2017_11_39_35_AM_A14_Q10_12095A.csv','Feat_8_10_2017_10_53_34_AM_A11_Q7_11617A.csv',
           'Feat_8_9_2017_5_13_20_PM_A08_Q5_10647A.csv', 'Feat_8_9_2017_4_07_50_PM_A07_Q4_10350A.csv', 
           'Feat_8_9_2017_3_37_20_PM_A05_Q3_10197A.csv', 'Feat_8_9_2017_2_25_13_PM_A04_Q2_9826A.csv']

print('Selected training files: ', type0files,type1files)
print('Selected test files: ', tefiles)


#######################################Load Train dataset#####################################
df0 = pd.DataFrame()
for f in type0files: 
    df = pd.read_csv((filedir +'/'+f),index_col=0)
    
    df.sort_values('location', ascending= True)     
    df['IMAG'] = scipy.signal.medfilt(df['IMAG'].values, 101)
    df['type'] = np.zeros(len(df['IMAG']))
    
    df = df[df['IMAG']>5000]
    print(len(df), ' of type 0')
    df = df.reset_index()  
    
    df = df.reindex(np.random.permutation(df.index))
    #df = df.iloc[:2860,:]
    df0 = pd.concat([df0,df], ignore_index=True)

#    plt.hist(df['IMAG'],50)

    
for f in type1files: 
    df = pd.read_csv((filedir +'/'+f),index_col=0)
    
    df.sort_values('location', ascending= True)     
    df['IMAG'] = scipy.signal.medfilt(df['IMAG'].values, 101)
    df['type'] = np.ones(len(df['IMAG']))
    
    df = df[df['IMAG']>5000] 
    df = df[df['IMAG']<14300]
    print(len(df), ' of type 1')
    
    df = df.reset_index()  
    df = df.reindex(np.random.permutation(df.index))
    
    #df = df.iloc[:2860,:]
    
    df0 = pd.concat([df0,df], ignore_index=True)    
#    plt.hist(df['IMAG'],50) 

#Normalize wavelet coefficients
df0['E'] = df0['D1']**2 + df0['D2']**2 + df0['D3']**2 + df0['D4']**2 + df0['D5']**2
df0['D1'] = df0['D1'] **2 / df0['E']
df0['D2'] = df0['D2'] **2 / df0['E']
df0['D3'] = df0['D3'] **2 / df0['E']
df0['D4'] = df0['D4'] **2 / df0['E']
df0['D5'] = df0['D5'] **2 / df0['E']

df0 = df0.reset_index()   
df0 = df0.reindex(np.random.permutation(df0.index))

#Define training dataset
train = df0



#############################################################################################


#Split the chosen training dataset it in two halves for testing
train_split = train[:(len(train)//2)]
trial_split = train[(len(train)//2):]

#plt.figure(2)
#plt.scatter(train.index, train['IMAG'], s=5)
#plt.show()

# Show the number of observations for the test and training dataframes
print()
print('Number of observations in the training data:', len(train_split))
print('Number of observations in the test dataset 1:',len(trial_split))
print()


# Create a list of the feature column's names, for both full and split datasets:
features = train.columns[8:13]
features_sp = train_split.columns[8:13]

# View features
print(features)
print()

#Define the parameter in split dataset to train against:
y_sp = train_split['type']

# Create a random forest classifier
cls1 = RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=None,bootstrap=True)

# Train classifier on the split training dataset
cls1.fit(train_split[features_sp], y_sp)

#Apply the classifier we trained to the test portion of the split dataset (which it has never seen before)
allpreds_split = cls1.predict(trial_split[features_sp])

print(pd.crosstab(trial_split['type'], allpreds_split, rownames=['Actual type'], colnames=['Predicted type']))
# View a list of the features and their importance scores
list(zip(trial_split[features_sp], cls1.feature_importances_))


##############################################################
#Now do same analysis for other quenches
#####################################################
#Train the classifier on the full training dataset
ctr =0 
y = train['type']
cls1.fit(train[features], y)

#Apply the classifier we trained to the test data (which it has never seen before)
ra = np.array([])
for tf in tefiles:
    
    df1 = pd.DataFrame()
    df1 = pd.read_csv((filedir + '/'+ tf),index_col=0)
    df1.sort_values('location', ascending= True)
    df1 = df1[df1['IMAG']<14000]
    df1 = df1[df1['IMAG']>500]
    
    #Normalize wavelet coefficients
    df1['E'] = df1['D1']**2 + df1['D2']**2 + df1['D3']**2 + df1['D4']**2 + df1['D5']**2
    df1['D1'] = df1['D1'] **2 / df1['E']
    df1['D2'] = df1['D2'] **2 / df1['E']
    df1['D3'] = df1['D3'] **2 / df1['E']
    df1['D4'] = df1['D4'] **2 / df1['E']
    df1['D5'] = df1['D5'] **2 / df1['E']

    
    df1 = df1.reset_index()
    df1 = df1.reindex(np.random.permutation(df1.index))
    test = df1    

    allpreds = cls1.predict(test[features])
    test['type'] = allpreds
    test_groups = test.groupby(['type'])
    test_type0 = test_groups.get_group((0))
    test_type1 = test_groups.get_group((1))
    #test_type2 = test_groups.get_group((2))
     
    ratio = len(test_type0) / ( len(test_type0) + len(test_type1) )
    ra = np.append(ra,ratio)
    ctr=ctr+1
    print(ctr)
print(ra)

plt.figure(3)
rcParams.update({'font.size': 20})
plt.scatter([103,79,64,49,35,25,11,10,7,5,4,3,2], ra, s=40)
plt.show()
