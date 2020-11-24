
#Import of libraries

import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from math import sqrt,pi,exp

#1.1
df = pd.read_csv("tabBats.csv",sep=',')

#1.2
print(df.head()) #describe the beginning of the dataset
print(df.shape) #give the size of the data matrix

#1.3
print("\n")
print("Average body mass:")
print(np.mean(df["BOW"]))
print("Body mass variance:")
print(np.var(df["BOW"]))
print("\n")
print("Standard deviation of the body mass:")
print(np.std(df["BOW"]))

print("\n")
print("Median brain mass:")
print(np.median(df["BRW"]))


plt.hist(df["MOB"],bins=7,color='red',rwidth=0.8)




#2
df2 = pd.read_csv("notes_fixed.csv",sep=';')
print(df2.head()) #describe the beginning of the dataset
print(df2.shape) #give the size of the data matrix


figure,axe = plt.subplots(figsize=(10,4))
plt.hist(df2["Lab1"],bins=range(0,20),color='red',rwidth=0.8)
plt.hist(df2["Lab3"],bins=range(0,20),color='red',rwidth=0.8)
plt.hist(df2["Lab7"],bins=range(0,20),color='red',rwidth=0.8)


unique, counts = np.unique(df2["GPA"], return_counts=True)
values = dict(zip(unique, counts));
plt.pie(values.values(), labels=values.keys())

#if we suppose that both follow a gaussian law, then you just need to compute the mean vlaues and standard deviation for both columns. Then use the IC95 formula for differences of mean.


#3

Mark = [6,8,9,10,11,12,13,14,17]
Number = [10,12,48,23,24,48,9,14,22]

#Then we plot these arrays, as an histogram
figure,axe = plt.subplots(figsize=(10,4))
plt.hist(Mark,bins=22,color='red',weights=Number,rwidth=0.8)
axe.set_title("Histogram of the dataset")
axe.set_xlabel("Mark")
axe.set_ylabel("Frequency")


totalmark=[a*b for a,b in zip(Mark,Number)]

mean = np.sum(totalmark)/np.sum(Number)


dispersion_mark_median = stats.binned_statistic(Number,Mark,statistic="median",bins=1)
median = dispersion_mark_median[0][0]

numvar = []
i=0
while i<len(Mark):
    numvar.append(Number[i]*(Mark[i]-mean)**2)
    i+=1

variance = sum(numvar)/sum(Number)

dispersion_mark_min = stats.binned_statistic(Number,Mark,statistic="min",bins=1)
min = dispersion_mark_min[0][0]


dispersion_mark_max = stats.binned_statistic(Number,Mark,statistic="max",bins=1)
max = dispersion_mark_max[0][0]


mode = Mark[np.argmax(Number)]


sumarray = np.dot(Mark,Number)



#We display everything

print("Min: {0} ".format(min))
print("Mean: {0} ".format(mean))
print("Variance: {0} ".format(variance))
print("Standard deviation: {0}  ".format(sqrt(variance)))
print("Median: {0} ".format(median))
print("Max: {0} ".format(max))
print("Mode: {0} ".format(mode))



#4
df3 = pd.read_csv("malnutrition.csv",header=None)
print(df3.head())

meanQImalnut=np.mean(df3)
stdQImalnut=np.std(df3)

print("\n")
print("Malnutrition sample size: {0} ".format(df3.size))
print("Mean QI with malnutrition: {0} ".format(meanQImalnut))
print("QI std with malnutrition: {0} ".format(stdQImalnut))


IC95minus=meanQImalnut-1.96*stdQImalnut/sqrt(df3.size)
IC95plus=meanQImalnut+1.96*stdQImalnut/sqrt(df3.size)

print("\n")
print("IC95 QI malnutrition: [{0};{1}] ".format(IC95minus,IC95plus))  #100 n'est pas dans l'intervalle à 95%, l'effet est donc significatif