
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
from scipy.stats import chi2_contingency

df = pd.read_csv("iris.csv")
print(df.describe().T)
print(df.head())

SepalLength=df['sepal_length']
SepalWidth=df['sepal_width']
PetalLength=df['petal_length']
PetalWidth=df['petal_width']

fig,axes = plt.subplots(2,2,figsize=(18,8))
sns.distplot(SepalLength,ax=axes[0][0])
sns.distplot(SepalWidth,ax=axes[0][1])
sns.distplot(PetalLength,ax=axes[1][0])
sns.distplot(PetalWidth,ax=axes[1][1])

sns.pairplot(df)

corr_matrix = df.corr()
print(corr_matrix.head())


fig,axes = plt.subplots(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,center=0)


def confidence_interval(r,n):
    Z = (np.log(1+r)-np.log(1-r))/2
    sz = np.sqrt(1/(n-3))
    Zinf = Z-1.96*sz
    Zsup = Z+1.96*sz
    icinf,icsup = ((np.exp(2*Zinf)-1)/(np.exp(2*Zinf)+1)),((np.exp(2*Zsup)-1)/(np.exp(2*Zsup)+1))
    return icinf,icsup

def compute_all_interval(dataframe):
    attributelist = list(dataframe.columns.values)
    n = np.square(len(attributelist))
    corr_matrix = dataframe.corr()
    interval_matrix_min = []
    interval_matrix_max = []
    
    for i in range(0,len(attributelist)):
        for j in range(0,len(attributelist)):
            interval_matrix_min.append(confidence_interval(corr_matrix.values[i][j],dataframe.shape[0])[0])
            interval_matrix_max.append(confidence_interval(corr_matrix.values[i][j],dataframe.shape[0])[1])
            
    leninterval = dataframe.shape[1]
    interval_matrix_min = np.reshape(interval_matrix_min,(leninterval,leninterval))
    interval_matrix_max = np.reshape(interval_matrix_max,(leninterval,leninterval))
    difference_interval = np.subtract(interval_matrix_max,interval_matrix_min)
    interval_matrix_min= pd.DataFrame(interval_matrix_min,dataframe.columns,columns=dataframe.columns);
    interval_matrix_max= pd.DataFrame(interval_matrix_max,dataframe.columns,columns=dataframe.columns);
    difference_interval_df = pd.DataFrame(difference_interval,dataframe.columns,columns=dataframe.columns);
    return interval_matrix_min,interval_matrix_max,difference_interval_df

iris_interval_min,iris_interval_max,iris_interval_diff = compute_all_interval(df)

fig,axes = plt.subplots(figsize=(10,8))
sns.heatmap(iris_interval_min,annot=True,center=0)
print('Minimum of the confidence interval of the iris dataset')

fig,axes = plt.subplots(figsize=(10,8))
sns.heatmap(iris_interval_max,annot=True,center=0)
print('Maximum of the confidence interval of the iris dataset')


df_mansize = pd.read_csv("mansize.csv",sep=";")
print(df_mansize.describe().T)

sns.pairplot(df_mansize)

corr_matrix2 = df_mansize.corr()
det_matrix2=corr_matrix2*corr_matrix2
print(corr_matrix2.head())


fig,axes = plt.subplots(figsize=(10,8))
sns.heatmap(corr_matrix2,annot=True,center=0)

inferval_mansize_min,inferval_mansize_max,inferval_mansize_diff = compute_all_interval(df_mansize)

fig,axes = plt.subplots(figsize=(15,10))
mask = np.zeros_like(inferval_mansize_diff)

with sns.axes_style("white"):
    ax = sns.heatmap(inferval_mansize_min, mask=mask,annot=True)
    print('Minimum of the interval confidence')
    
 
df_weather = pd.read_csv("weather.csv",sep=";")    
df_weather = df_weather.drop(['City'],axis=1)  #we don't need the cities
print(df_weather.head())

outlooknbval = df_weather['Outlook'].value_counts()
humiditynbval = df_weather['Humidity'].value_counts()
temperaturenbval = df_weather['Temperature'].value_counts()

fig,axes = plt.subplots(2,2,figsize=(18,15))
outlooknbval.plot(kind='bar',ax=axes[0][0],title="Outlook")
humiditynbval.plot(kind='bar',ax=axes[0][1],title="Humidity")
temperaturenbval.plot(kind='bar',ax=axes[1][0],title="Temparature")

fig,axes = plt.subplots(1,3,figsize=(18,8))
# Plotting the occurence of each temperature values depending on other attributes
sns.countplot(x="Temperature",data=df_weather,ax=axes[0],palette="Blues_d")
sns.countplot(x="Temperature",data=df_weather,hue='Temperature',palette="Blues_d")
sns.countplot(x="Temperature",data=df_weather,hue='Outlook',palette="Blues_d",ax=axes[1]) 

fig,axes = plt.subplots(1,3,figsize=(18,8))
# Plotting the occurence of each Te values depending on other attributes
sns.countplot(x="Outlook",data=df_weather,ax=axes[0],palette="Blues_d")
sns.countplot(x="Outlook",data=df_weather,hue='Temperature',palette="Blues_d")
sns.countplot(x="Outlook",data=df_weather,hue='Humidity',palette="Blues_d",ax=axes[1])   

fig,axes = plt.subplots(1,3,figsize=(18,8))
# Plotting the occurence of each humidity values depending on other attributes
sns.countplot(x="Humidity",data=df_weather,ax=axes[0],palette="Blues_d")
sns.countplot(x="Humidity",data=df_weather,hue='Temperature',palette="Blues_d")
sns.countplot(x="Humidity",data=df_weather,hue='Outlook',palette="Blues_d",ax=axes[1])

crosstab = pd.crosstab(df_weather['Outlook'],df_weather['Temperature'])
print(crosstab.head())

def degre_de_liberte(df):
    return((df.shape[0]-1)*(df.shape[1]-1))
    
print("Degré de liberté", degre_de_liberte(crosstab))


chi2, p, dof, ex = chi2_contingency(crosstab)
df_contagency = pd.DataFrame(ex,index="Foggy,Overcast,Rainy,Sunny".split(","),
columns="Cold,Hot,Mild".split(','))

print(df_contagency)


def p_values(Index1,Index2,df):
    df_result = pd.crosstab(df_weather[Index1],df[Index2])
    chi2, p, dof, ex = chi2_contingency(df_result)
    return p

def cramers(Index1,Index2,df):
    df_result = pd.crosstab(df[Index1],df[Index2])
    chi2, p, dof, ex = chi2_contingency(df_result)
    n = len(df[Index1])
    r=df_result.shape[0]
    c=len(df_result.columns)
    n=0
    for i in range(0,c):
        for j in range(0, r):
            n=n+df_result[df_result.columns[i]][j]
            
    cramer=(chi2/(n*(min((c-1),(r-1)))**1))**0.5
    return cramer

print("Pvalue for Outlook and Temperature",p_values("Outlook","Temperature",df_weather))
print("Pvalue for Outlook and Humidity",p_values("Outlook","Humidity",df_weather))
print("Pvalue for Temperature and Humidity",p_values("Temperature","Humidity",df_weather))

print("Cramer's coefficient for Outlook and Humidity :", cramers("Outlook", "Humidity",df_weather))
