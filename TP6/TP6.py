import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statistics import pstdev 
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import seaborn as sns
import statsmodels.tsa.stattools as ts

# =============================================================================
# A.1
# =============================================================================

x = np.random.normal(loc = 0, scale = 1, size = 1000)
print(x)
s = pd.Series(x)
print(s)


# =============================================================================
# A.2
# =============================================================================
plt.title("Histogramme du bruit blanc gaussien contenant 1000 échantillons")
sns.distplot(x, hist=True)
plt.show()

plt.title("Bruit blanc gaussien contenant 1000 échantillons")
s.plot(figsize=(10,4))
plt.show()

# =============================================================================
# A.3
# =============================================================================
f = plt.figure(figsize=(12,8))
ax1 = f.add_subplot(211)
f = sm.graphics.tsa.plot_acf(s, lags=20, ax=ax1)
ax2 = f.add_subplot(212)
f = sm.graphics.tsa.plot_pacf(s, lags=20, ax=ax2)

# =============================================================================
# #A.4
# =============================================================================

#Test ADF
result = ts.adfuller(s, 1)
print(result)

def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = ts.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
adf_test(s)

#test KPSS

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = ts.kpss(timeseries, nlags=20)
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
    
kpss_test(x)


# =============================================================================
# B.1
# =============================================================================
sales = pd.read_csv("sales.csv")
print(sales.head())

food = sales['Food'][0:48]
print(food)

plt.title("Histogramme du chiffre d'affaire produits alimentaires ")
sns.distplot(food, hist=True)
plt.show()

plt.title("chiffre d'affaire produits alimentaires")
food.plot(figsize=(10,4))
plt.show()


fuel = sales['Fuel'][0:48]

plt.title("Histogramme du chiffre d'affaire de carburant ")
sns.distplot(fuel, hist=True)
plt.show()

plt.title("chiffre d'affaire d'affaire de carburant")
fuel.plot(figsize=(10,4))
plt.show()

# Test Shapiro-Wilk

shapiro_test_food = stats.shapiro(food)
print(shapiro_test_food)

shapiro_test_fuel = stats.shapiro(fuel)
print(shapiro_test_fuel)

# Test de Box-Pierce

res = sm.tsa.ARMA(food, (1,1)).fit(disp=-1)
sm.stats.acorr_ljungbox(res.resid, return_df=True)
sm.stats.acorr_ljungbox(food, return_df=True)


res = sm.tsa.ARMA(fuel, (1,1)).fit(disp=-1)
sm.stats.acorr_ljungbox(res.resid, return_df=True)
sm.stats.acorr_ljungbox(fuel, return_df=True)

# =============================================================================
# B.2
# =============================================================================

#1
fuel_d = fuel.diff()
print(fuel_d)
fuel_d = fuel_d[1:]
print(fuel_d)


#2
plt.title("diffrérentiation t - t+1 chiffre d'affaire de carburant ")
sns.distplot(fuel_d, hist=False)
plt.show()

#3
f = plt.figure(figsize=(12,8))
ax1 = f.add_subplot(211)
f = sm.graphics.tsa.plot_acf(x, lags=20, ax=ax1)
ax2 = f.add_subplot(212)
f = sm.graphics.tsa.plot_pacf(x, lags=20, ax=ax2)

#4
adf_test(fuel_d)

kpss_test(fuel_d)

df = pd.read_csv("sales.csv")
#print(df)

# =============================================================================
# C
# =============================================================================

#1
xtrain = []
for i in range(48):
    xtrain.append(df['Fuel'][i])
print(xtrain)

xtest = []
for i in range(48, 50):
    xtest.append(df['Fuel'][i])
print(xtest)

#2
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(xtrain, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(xtrain, lags=40, ax=ax2)

#3
print(ARIMA(xtrain, order=(1, 2, 1)).fit().summary())
print("BIC")
print(ARIMA(xtrain, order=(1, 2, 1)).fit().bic)
print("AIC")
print(ARIMA(xtrain, order=(1, 2, 1)).fit().aic)
Erreur_type = pstdev(xtrain) 
print(Erreur_type)

print(ARIMA(xtrain, order=(1, 2, 2)).fit().summary())
print("BIC")
print(ARIMA(xtrain, order=(1, 2, 2)).fit().bic)
print("AIC")
print(ARIMA(xtrain, order=(1, 2, 2)).fit().aic)

print(ARIMA(xtrain, order=(8, 2, 2)).fit().summary())
print("BIC")
print(ARIMA(xtrain, order=(8, 2, 2)).fit().bic)
print("AIC")
print(ARIMA(xtrain, order=(8, 2, 2)).fit().aic)

#4 - Modele 1 et 2

#5 
residuals = DataFrame(ARIMA(xtrain, order=(1, 2, 1)).fit().resid)

residuals2 = DataFrame(ARIMA(xtrain, order=(1, 2, 2)).fit().resid)

residuals3 = DataFrame(ARIMA(xtrain, order=(8, 2, 2)).fit().resid)

# Test Shapiro-Wilk
print(stats.shapiro(residuals))

print(stats.shapiro(residuals2))

print(stats.shapiro(residuals3))

# Test de Box-Pierce
sm.stats.acorr_ljungbox(residuals, return_df=True)
sm.stats.acorr_ljungbox(xtrain, return_df=True)

sm.stats.acorr_ljungbox(residuals2, return_df=True)
sm.stats.acorr_ljungbox(xtrain, return_df=True)

sm.stats.acorr_ljungbox(residuals3, return_df=True)
sm.stats.acorr_ljungbox(xtrain, return_df=True)
                        
# Test ADF et KPSS
def ADFKPSS(donnees):
    ADF = adfuller(donnees)   
    print( 'ADF Statistic: {}'.format( ADF[0] ) )
    print( 'p-value: {}'.format( ADF[1] ) )
    print( 'Critical Values:' )
    for key, value in ADF[4].items():
        print( '{}: {}'.format(key, value) )        
    print()   
    KPSS = kpss(donnees, nlags='auto')
    print('KPSS Statistic: {}'.format(KPSS[0]))
    print( 'p-value : {}'.format(KPSS[1]))
    print( 'Critical Values:' )
    for key, value in KPSS[3].items():
        print( '{}: {}​​​​​​​​'.format(key, value) )
    print()
ADFKPSS(residuals)

#6
model = ARIMA(xtrain, order=(1, 0, 1))
#Train the ARMA model
model_fit = model.fit(disp=0)
#Test the model
x_pred = model_fit.forecast(steps=len(xtest))[0]
#Compare visually the predicted and true sale vales
plt.figure();
model_fit.plot_predict(1, len(xtrain) + len(xtest));
plt.plot(range(len(xtrain)+1, len(xtrain)+1 + len(xtest)), xtest,
label="test");
plt.title("ARIMA(1, 0, 1) sur les ventes de carburant");
plt.legend();
plt.show();


print(model)
#erreur quadratique
print('mse (sklearn): ', mean_squared_error(xtest,xtest))


