#!/usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
from seaborn import countplot


formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(description='KMEANS Model application', formatter_class=formatter)

required = parser.add_argument_group(title='required')
required.add_argument('file', type=str, help='data file to read')
args = parser.parse_args()
file_name = args.file
file_path=  os.path.abspath(file_name)

#file_path = "C:\\Users\\Heath\\OneDrive\\Documents\\CS\\CS479\\FinalProject\\cleaned_crime_data_2021_modified_V1.csv"

#read csv into pandas dataframe
data = pd.read_csv(file_path)
data.head(10)

#drop driver's license and area name columns as they will not be targeted.
data = data.drop(columns=['DR_NO','AREA.NAME'])
data['DATE.OCC'] = data['DATE.OCC'].str.strip(' 0:00')

def convertDateTime(date):
    return datetime.strptime(date, '%m/%d/%Y').timestamp()

data['DATE.OCC'] = data['DATE.OCC'].apply(convertDateTime)
#data.head(10)

#In this current state, not all data is machine readable.
#Vict.Sex and Vict.Descent both need to be converted to a machine readable format.
#We will use one-hot encoding to do this.
#Before we can encode, we will to reduce the number of categories for both Vict.Sex and Vict.Descent
#This is done because a majority of the categories aren't used and will only complicate things later on if they aren't removed
#As seen in our EDA:
    #the 'H' category only occurs 28 times in Vict.Sex, so we will discard it to reduce the dimensions of our one-hot vectors
    #over 90% of the Vict.Descent data lies in 5 descents, the rest we can discard to reduce the dimensions further

#Remove any rows with H in their Vict.Sex column
encode = data.drop(data[data['Vict.Sex']=='H'].index)

#Remove any rows without the specified characters below in their Vict.Descent column
toKeep = ['B','H','O','W','X']
encode = encode.drop(encode[encode['Vict.Descent'].isin(toKeep)==False].index)
#encode.head(10)
#the table printed now only stores records with the most meaningful data

#here, we will implement Kmodes (similar to KMeans, but clusters categorical variables rather than numerical). 
#to do this, for now, we will strip the date and loc (lat, lon) data. What's left is our categorical data
#but we will categorize age data in ranges (10-20, 21-30, 31-40, etc.) as well as time data

KmodesData = pd.DataFrame.copy(encode)
KmodesData.head()
KmodesData.drop(columns=['DATE.OCC', 'LAT', 'LON'], inplace=True)

#group ages
KmodesData['AgeBins'] = pd.cut(KmodesData['Vict.Age'], bins=[0,20,30,40,50,60,70,80,max(KmodesData['Vict.Age'])])
KmodesData.drop(columns=['Vict.Age'], inplace=True)

#group times
KmodesData['TimeOccBins'] = pd.cut(KmodesData['TIME.OCC'], bins=[0, 600, 1200, 1600, 2100, 2400])
KmodesData.drop(columns=['TIME.OCC'], inplace=True)

#convert it all to strings to ensure categories can be determined by KModes (this may be unnecssary, just want to be safe)
KmodesData = KmodesData.astype('str', copy=True)
#KmodesData.head(10)

#Kmodes requires us to give it the number of clusters we wish to categorize
#we will use the Elbow method to determine this number of clusters K

cost = []
K = [1,2,3,4]
for i in K:
    kout = KModes(n_clusters=i, init='Cao', n_init=4)
    kout.fit_predict(KmodesData)
    cost.append(kout.cost_)
plt.plot(K, cost)
plt.xlabel('K')
plt.ylabel('Cost')
plt.show()

#we will select the farthest right bend as possible...
#we can see the bend at K=2, so we will use 2 clusters
#it also makes sense to use 2 clusters as we have a feature (Gender) that has two primary categories


#we will now implement oiur KModes algo
kout = KModes(n_clusters=2, init='Cao', n_init=4)

#and use the fitted model to assign clusters to each victim
clusters = kout.fit_predict(KmodesData)

#finally append the cluster values to our dataframe
KmodesData['Cluster'] = clusters
KmodesData.head(10)

#here, we will use countplots to visualize each feature's counts.
#again, we have two clusters so that is represented on the X axis with counts on the Y axis
#think of each cluster being a sample victim that is targeted most frequently
#we can see in the Vict.Sex plot, that cluster 0 is the males cluster and cluster 1 is the females cluster
#Looking at the rest of the diagrams, we can essentially determine the most commonly victimized male (0) and female (1)
#using these clusters, we can pick the largest value in each plot to "build" an at-risk victim profile

#Victim 1 (culster 0): Male, Hispanic, 30-40 years old, attacked between 4:00pm and 9:00pm
#Victim 2 (cluster 1): Female, White, 20-30 years old, attackd between 6:00am and 12:00pm
#We can say people with these characteristics are most likely to be attacked in LA

#we can add location data to give a potential area where they may be attacked.
#to do this, we will need to convert the lat and lon data to categorical data
#or use k-prototypes that allows for both numerical and categorical data features

for column in KmodesData.iloc[:,:-1]:
    plt.subplots()
    countplot(x='Cluster', hue=column, data=KmodesData)
    plt.show()

