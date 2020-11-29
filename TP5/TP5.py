import matplotlib as plt

import pandas as pd
import seaborn as sns
import random as rd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pydotplus
import collections
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn import tree
import random

#EXERCICE A.1

#1

titanic_train = pd.read_csv('titanic_train.csv')
print(titanic_train.head())

titanic_test = pd.read_csv('titanic_test.csv')
print(titanic_test.head())


titanic_train.info()

titanic_train.isnull().sum()

# ajout des ages manquants
fig, ax = plt.subplots();
sns.boxplot(x='Parch',y='Age',data=titanic_train)
plt.title("Repartition des ages en fonction de Parch")


def fill_age(cols):
    Parch = cols[1]
    Age =cols[0]
    if pd.isnull(Age):
        if Parch==0:
            return random.randint(25, 40)
        elif Parch==1:
            return random.randint(8, 35)
        elif Parch==2:
            return random.randint(10, 27)
        elif Parch==3:
            return random.randint(25, 50)
        elif Parch==4:
            return random.randint(35, 53)
        elif Parch==5:
            return 40
        elif Parch==6:
            return 44
        else:
            return titanic_train.fillna('ffill')
    else:
        return Age

titanic_train['Age']=titanic_train[["Age","Parch"]].apply(fill_age,axis=1)


titanic_train['Age'].isnull().sum()





#2


fig, ax = plt.subplots();
sns.set_style('whitegrid')
sns.distplot(titanic_train.Age, kde=False,bins=40,color='g')
plt.title("Histogramme des âges sans les valeurs manquantes")


print("Pourcentage de personnes décédés: ", titanic_train['Survived'].value_counts()[0]/titanic_train['Survived'].shape[0]*100, "%")
pd.crosstab(titanic_train['Survived'], titanic_train['Survived'], normalize=True)


titanic_train['Pclass'].value_counts()

n1 = titanic_train['Pclass'].value_counts()[1]
n2 = titanic_train['Pclass'].value_counts()[2]
n3 = titanic_train['Pclass'].value_counts()[3]
x = [n1,n2,n3]

fig, ax = plt.subplots();
plt.pie(x, labels = ['1ère classe', '2ème classe', '3ème classe'], autopct='%1.1f%%')
plt.title("Diagramme 'camembert' du nombre de passagers par classe")



titanic_train['Sex'].value_counts(normalize = True)

pd.crosstab(titanic_train['Sex'], titanic_train['Survived'], normalize = True)

titanic_train[['Sex','Survived']].groupby(['Sex']).mean()



# - 
# ajout d'une nouvelle colonne pour child/not child
l = []
for i in titanic_train.index:
    if titanic_train['Age'][i] < 18:
        l.append(1)
    elif titanic_train['Age'][i] >= 18:
        l.append(0)


len(l)
   
titanic_train['Child'] = l  
print(titanic_train['Child'][5])

titanic_train[['Child','Survived']].groupby(['Child']).mean()



titanic_train[['Sex','Child', 'Survived']].groupby(['Sex', 'Child']).mean()


#- 
pd.crosstab(titanic_train['Pclass'], titanic_train['Survived'])

titanic_train[['Pclass','Survived']].groupby(['Pclass']).mean()

fig, ax = plt.subplots();
sns.countplot(x='Survived', data=titanic_train, palette='rainbow',hue='Pclass')



# 4 

corr = titanic_train.corr()

fig, ax = plt.subplots();
sns.heatmap(corr, 
     xticklabels=corr.columns, 
     yticklabels=corr.columns) 
plt.title("Matrice de correlation des differents attributs")


# 5


lFare2= []
for i in titanic_train.index:
    if titanic_train['Fare'][i] < 10:
        lFare2.append(1)
    elif titanic_train['Fare'][i] < 20:
        lFare2.append(2)
    elif titanic_train['Fare'][i] < 30:
        lFare2.append(3)
    elif titanic_train['Fare'][i] >= 30:
        lFare2.append(4)


len(lFare2)
   
titanic_train['Fare2'] = lFare2

print(titanic_train.head())


titanic_train['Fare2'].corr(titanic_train['Pclass'])

titanic_train[['Fare2', 'Survived']].groupby(['Fare2']).mean()

pd.crosstab(titanic_train['Fare2'], titanic_train['Survived'])



#EXERCICE A.2

#1


"""
Travail effectué dans l'exercice 1
def fill_age(cols):
    Parch = cols[1]
    Age =cols[0]
    if pd.isnull(Age):
        if Parch==0:
            return rd.randint(25, 40)
        elif Parch==1:
            return rd.randint(8, 35)
        elif Parch==2:
            return rd.randint(10, 27)
        elif Parch==3:
            return rd.randint(25, 50)
        elif Parch==4:
            return rd.randint(35, 53)
        elif Parch==5:
            return 40
        elif Parch==6:
            return 44
        else:
            return titanic_train.fillna('ffill')
    else:
        return Age
titanic_train['Age']=titanic_train[["Age","Parch"]].apply(fill_age,axis=1)

# ajout d'une nouvelle colonne pour child/not child
l = []
for i in titanic_train.index:
    if titanic_train['Age'][i] < 18:
        l.append(1)
    elif titanic_train['Age'][i] >= 18:
        l.append(0)
len(l)
#implémentation   
titanic_train['Child'] = l  
print(titanic_train.head()) """

#2

# ajout d'une nouvelle colonne pour Sex_Cat
S = []
for i in titanic_train.index:
    if titanic_train['Survived'][i] == 1:
        S.append(1)

len(S)
#implémentation 582  

print(len(S)/582)

# ajout d'une nouvelle colonne pour Sex_Cat
Sex_categorie = []
for i in titanic_train.index:
    if titanic_train['Sex'][i] == 'male':
        Sex_categorie.append(1)
    elif titanic_train['Sex'][i] == 'female':
        Sex_categorie.append(0)
len(Sex_categorie)
#implémentation   
titanic_train['Sex_Cat'] = Sex_categorie  
print(titanic_train.head())


#classifier choice

gnbModel=GaussianNB()

#choice of the training set, considered attributes and variable to decrease predict

gnbModel.fit(titanic_train[['Child', 'Sex_Cat']], titanic_train['Survived'])

#Sex_cat stands for transformed categorical array
#expected results  are stored in a separate vector
expected = titanic_train['Survived']

#predictions on the training set
predicted = gnbModel.predict(titanic_train[['Child','Sex_Cat']])

#displaying relevant metrics
print(metrics.classification_report(expected, predicted))

#same when applying the model to the test set
expected = titanic_train['Survived']
predicted = gnbModel.predict(titanic_train[['Child', 'Sex_Cat']])
print(metrics.classification_report(expected, predicted))


#3
#classifier choice
gnbModel=GaussianNB()

#choice of the training set, considered attributes and variable to decrease predict
gnbModel.fit(titanic_train[['Child', 'Sex_Cat', 'Fare2']], titanic_train['Survived'])

#Sex_cat stands for transformed categorical array
#expected results  are stored in a separate vector
expected = titanic_train['Survived']

#predictions on the training set
predicted = gnbModel.predict(titanic_train[['Child','Sex_Cat', 'Fare2']])

#displaying relevant metrics
print(metrics.classification_report(expected, predicted))

#same when applying the model to the test set
expected = titanic_train['Survived']
predicted = gnbModel.predict(titanic_train[['Child', 'Sex_Cat', 'Fare2']])
print(metrics.classification_report(expected, predicted))

#EXERCICE A.3

Sex_categorie = []
for i in titanic_train.index:
    if titanic_train['Sex'][i] == 'male':
        Sex_categorie.append(1)
    elif titanic_train['Sex'][i] == 'female':
        Sex_categorie.append(0)
len(Sex_categorie)
#implémentation   
titanic_train['Sex_Cat'] = Sex_categorie 

clf = tree.DecisionTreeClassifier()
clf = clf.fit(titanic_train[['Child','Pclass', 'Sex_Cat']], titanic_train['Survived'])

fig, ax = plt.subplots();
tree.plot_tree(clf) 


dot_data = tree.export_graphviz(clf , out_file=None , 
                                feature_names=['Child','Pclass','Sex_Cat'],
                                class_names=['Die', 'Survived'], filled=True,rounded=True,special_characters=True, precision=0) 
graph = pydotplus.graph_from_dot_data(dot_data)  


# Create PNG
graph.write_png("test.png")




