#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') 


# In[79]:


original_data = pd.read_csv(r"project 177 final data set for model bulting .csv")


# In[80]:


original_data.head()


# In[81]:


df = original_data.copy()


# In[82]:


df['Date'] = pd.to_datetime(df['Date'])


# In[83]:


df['Year'] = df['Date'].dt.year


# In[ ]:





# In[84]:


df.set_index('Date', inplace = True) 


# In[85]:


print(df.info())


# In[86]:


df.describe().T


# In[92]:


df['Ferro_Nickel']=df[' Ferro Nickel']
df.drop(' Ferro Nickel',axis=1, inplace = True)


# In[88]:


df.isnull().sum()


# we have some null values 
# we will treat them accprdingly later

# Now we will check for the the diplicated values present in the data set

# In[89]:


df.duplicated().sum()


# Clealy we dont have any dulplicate values

# Now we will check for some of the basic (statistical) information or description

# In[90]:


df.describe().T


# In[ ]:





# # First moment Business Decision

# In[ ]:





# In[102]:


yearly_statistics = df.groupby('Year').agg({
    'Ferro_Nickel': ['mean', 'median', 'min', 'max'],
    'Aluminium': ['mean', 'median', 'min', 'max'],
    'Fluorite': ['mean', 'median', 'min', 'max'],
    'Graphite': ['mean', 'median', 'min', 'max'],
    'Manganese': ['mean', 'median', 'min', 'max'],
    'Molybdenum': ['mean', 'median', 'min', 'max'],
    'Vanadium': ['mean', 'median', 'min', 'max']
})

# Display the yearly statistics
print("Yearly Statistics:")
yearly_statistics.stack()


# # Second Moment Decision
# ### (VARIENCE, STADARD DEVIARION, RANGE)

# In[118]:


yearly_statistics = df.groupby('Year').agg({
    'Ferro_Nickel': ['std', 'var', lambda x: x.max() - x.min()],
    'Aluminium': ['std', 'var', lambda x: x.max() - x.min()],
    'Fluorite': ['std', 'var', lambda x: x.max() - x.min()],
    'Graphite': ['std', 'var', lambda x: x.max() - x.min()],
    'Manganese': ['std', 'var', lambda x: x.max() - x.min()],
    'Molybdenum': ['std', 'var', lambda x: x.max() - x.min()],
    'Vanadium': ['std', 'var', lambda x: x.max() - x.min() ]
})


# Display the yearly statistics
print("Yearly Statistics:")
print(yearly_statistics.stack())

df.agg({   'Ferro_Nickel': ['std', 'var', lambda x: x.max() - x.min()],
    'Aluminium': ['std', 'var', lambda x: x.max() - x.min()],
    'Fluorite': ['std', 'var', lambda x: x.max() - x.min()],
    'Graphite': ['std', 'var', lambda x: x.max() - x.min()],
    'Manganese': ['std', 'var', lambda x: x.max() - x.min()],
    'Molybdenum': ['std', 'var', lambda x: x.max() - x.min()],
    'Vanadium': ['std', 'var', lambda x: x.max() - x.min() ]
})


# # skewness and kurtosis:-
# #### Skewness measures the asymmetry of the distribution.
# #### Kurtosis measures the tailedness of the distribution.
# 

# In[130]:


from scipy.stats import skew, kurtosis

# Define skewness and kurtosis functions
def compute_skewness(series):
    return skew(series.dropna())

def compute_kurtosis(series):
    return kurtosis(series.dropna())

# Compute yearly statistics
yearly_statistics = df.groupby('Year').agg({
    'Ferro_Nickel': [compute_skewness, compute_kurtosis],
    'Aluminium': [compute_skewness, compute_kurtosis],
    'Fluorite': [compute_skewness, compute_kurtosis],
    'Graphite': [compute_skewness, compute_kurtosis],
    'Manganese': [compute_skewness, compute_kurtosis],
    'Molybdenum': [compute_skewness, compute_kurtosis],
    'Vanadium': [compute_skewness, compute_kurtosis],
})

# Print yearly statistics
print("Yearly Statistics:")
print(yearly_statistics.stack())




# # Graphs

# In[137]:


fig, axs = plt.subplots(7,3,figsize=(15,30))
axs = axs.flatten()

metal_columns = ['Ferro_Nickel', 'Aluminium', 'Fluorite', 'Graphite', 'Manganese', 'Molybdenum', 'Vanadium']

for i, metal in enumerate(metal_columns):
  
    sns.boxplot(y=df[metal], data=df,ax = axs[3*i])
    axs[3*i].set_title(f'Box Plot for {metal}')
    axs[3*i].set_xlabel('')
        
    sns.violinplot( y=df[metal], data=df, ax = axs[3*i+1])
    axs[3*i+1].set_title(f'Violin Plot for {metal}')
    axs[3*i+1].set_xlabel('')

    sns.histplot(df[metal],ax = axs[3*i+2],bins=20 , kde = True)
    axs[3*i+2].set_title(f'Hist Plot for {metal}')
    axs[3*i+2].set_xlabel('')

plt.tight_layout()
plt.show()


# In[147]:


# Filter data for the desired date range (from January 2020 to December 2023)
df_date_range = df[(df.index >= '2020-01-01') & (df.index <= '2023-12-31')]

# Resample the data monthly and aggregate using mean
df_monthly = df_date_range.resample('M').mean()

# Plot separate line chart for each metal column
for metal in metal_columns:
    plt.figure(figsize=(25, 12))
    plt.plot(df_monthly.index, df_monthly[metal])
    plt.title(f'{metal} Prices from 2020 to 2023 (Monthly)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.show()


# inputing missing values now

# In[11]:


from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)
for column in numerical_columns:
    df[column] = knn_imputer.fit_transform(df[[column]])
    


# In[12]:


df.isnull().sum()


# In[13]:


df.head(50)


# In[14]:


# from statsmodels.tsa.seasonal import seasonal_decompose

# result = seasonal_decompose(df[' Ferro Nickel'], model = 'additive', period= 12)

# trend = result.trend

# seasonal = result.seasonal

# residual = result.resid

# plt.figure(figsize =(14,10))

# plt.subplot(411)
# plt.plot(df[' Ferro Nickel'], label = 'original')
# plt.legend()

# plt.subplot(412)
# plt.plot(trend, label = 'Trend')
# plt.legend()
# 
# plt.subplot(413)
# plt.plot(seasonal, label = 'Seasonal')
# plt.legend()

# plt.subplot(414)
# plt.plot(residual, label = 'Residual')
# plt.legend()

# plt.tight_layout()
# plt.show()


# df['Difference_f_nickel']=df[' Ferro Nickel'].diff()

# plt.figure(figsize=(25,10))
# plt.plot(df['Date'], df[' Ferro Nickel'], label='Ferro Nickel')
# plt.plot(df['Date'], df['Difference_f_nickel'],label='Difference_f_nickel', linestyle = '--')
# plt.show()


# In[15]:


# df_year = df.resample('Y').mean()
# df_year


# In[16]:


# df_month = df.resample('m').mean()
# df_month


# In[17]:


# df_growth = df.resample('y').apply(lambda x:(x.iloc[0]-x.iloc[11])/(x.iloc[0])*100)
# df_growth


# In[18]:


# df_quater = df.resample('q').mean()
# df_quater


# In[ ]:





# In[19]:


df.set_index('Date',inplace = True)


# In[20]:


df.head()


# # AR

# In[21]:


from statsmodels.tsa.ar_model import AutoReg

x= df['Aluminium'].values

x.shape


x

from statsmodels.tsa.stattools import adfuller
dftest = adfuller(df['Aluminium'],autolag='AIC')

print("ADF Statistic:", dftest[0])
print("p-value:", dftest[1])
print("Num Of lags",dftest[2])
print('Num of observations used for ADF regression and critical values calculation',dftest[3])
print("Critical Values:")

for key, val in dftest[4].items():
    print("\t",key,": ", val)

from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
pacf=plot_pacf(df['Aluminium'], lags=5)
acf = plot_acf(df['Aluminium'],lags=5)






# In[22]:


train = df['Aluminium'].iloc[:42]
test= df['Aluminium'].iloc[42:]

model = AutoReg(train, lags=3).fit()

model.summary()


# In[23]:


plt.plot(train , label = 'Train')
plt.plot(test , label = 'Test')


# In[24]:


pred  = model.predict(start=len(train),end=len(x)-1, dynamic = False)

plt.plot(pred,label="Predicted")
plt.plot(test,color ='r',label='Actual')
plt.legend()


# In[ ]:





# In[ ]:





# In[25]:


from math import sqrt
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(test, pred))
rmse

pred_future = model.predict(start=len(train+test)+1, end = len(train + test)+7,dynamic = False)
print("The future prediction for next")
print(pred_future)
print("Number of prediction made \t", len(pred_future))


# In[26]:


plt.plot(train , label = 'Train')
plt.plot(test , label = 'Test')
plt.plot(pred_future , label = 'Prediction')
plt.show()


# In[ ]:





# # SARIMA

# In[27]:


import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# In[28]:


plt.plot(df['Molybdenum'])
plt.title('Molybdenum')
plt.show()


# In[229]:


molyb= df['Molybdenum']

molyb.shape, molyb


# In[230]:


from statsmodels.tsa.stattools import adfuller
dftest = adfuller(df['Molybdenum'],autolag='AIC')

print("ADF Statistic:", dftest[0])
print("p-value:", dftest[1])
print("Num Of lags",dftest[2])
print('Num of observations used for ADF regression and critical values calculation',dftest[3])
print("Critical Values:")

for key, val in dftest[4].items():
    print("\t",key,": ", val)

from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
pacf=plot_pacf(df['Molybdenum'], lags=5)
acf = plot_acf(df['Molybdenum'],lags=5)






# In[231]:


data_diff_arma = molyb.diff()


# In[232]:


data_diff_arma.dropna()


# In[233]:


plt.plot(data_diff_arma)
plt.title('Differenced Time Series Data')
plt.xlabel('Date')
plt.ylabel('molybdenum')
plt.xticks(rotation =  90)
plt.show()


# In[234]:


from pmdarima import auto_arima

# Automatic selection using auto_arima
stepwise_fit = auto_arima(df['Molybdenum'], seasonal=True, m=12,  # Adjust 'm' for the seasonality frequency
                          trace=True, suppress_warnings=True)

# Display the summary of the best fitted model
print(stepwise_fit.summary())


# In[235]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
train=df['Molybdenum'].iloc[:42]
test=df['Molybdenum'].iloc[42:]
train, test


# In[236]:


plt.plot(train, label='train')
plt.legend()


# In[237]:


plt.plot(test, label='test', color='r')
plt.legend()


# In[238]:


sari_model = SARIMAX(df['Molybdenum'], order=(0,1,1), seasonal_order=(0,1,1,12))
result = sari_model.fit()


# In[239]:


sari_forecast = result.get_forecast(steps=6)
forecast_values = sari_forecast.predicted_mean
confidence_intervals = sari_forecast.conf_int()

# Print forecasted values and confidence intervals
print("Forecasted Values:")
print(forecast_values)
print("\nConfidence Intervals:")
print(confidence_intervals)


# In[240]:


plt.plot(train)
plt.plot(test, color='r')
plt.plot(forecast_values, color='g')
plt.plot(confidence_intervals,color='orange',linestyle ='--')
plt.xticks(rotation = 90)


# In[ ]:





# In[ ]:





# # arima

# In[241]:


from statsmodels.tsa.ar_model import AutoReg

x= df['Manganese'].values

x.shape


x

from statsmodels.tsa.stattools import adfuller
dftest = adfuller(df['Manganese'],autolag='AIC')

print("ADF Statistic:", dftest[0])
print("p-value:", dftest[1])
print("Num Of lags",dftest[2])
print('Num of observations used for ADF regression and critical values calculation',dftest[3])
print("Critical Values:")

for key, val in dftest[4].items():
    print("\t",key,": ", val)

from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
pacf=plot_pacf(df['Manganese'], lags=5)
acf = plot_acf(df['Manganese'],lags=5)


# In[242]:


from pmdarima import auto_arima


# In[243]:


stepwise_fit =  auto_arima(df['Manganese'], trace =True, 
                           suppress_warnings =True)
stepwise_fit.summary()


# In[244]:


from statsmodels.tsa.arima_model import ARIMA


# In[245]:


train = df['Manganese'].iloc[:-7]
test=df['Manganese'].iloc[-7:]
print(train,test)


# In[246]:


pred  = model.predict(start=len(train),end=len(x)-1, dynamic = False)

plt.plot(pred,label="Predicted")
plt.plot(test,color ='r',label='Actual')
plt.legend()


# In[247]:


from math import sqrt
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(test, pred))
rmse

pred_future = model.predict(start=len(x)+1, end = len(x)+7,dynamic = False)
print("The future prediction for next")
print(pred_future)
print("Number of prediction made \t", len(pred_future))


# # ETS

# In[28]:


import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose


# In[29]:


# Ensure df['Molybdenum'] is a pandas Series
Vanadium_series = df['Vanadium']

# Perform seasonal decomposition
result = seasonal_decompose(Vanadium_series, model='additive', period=18)

# Plot the decomposed components
result.plot()


# In[30]:


train = df['Vanadium'].iloc[:42]
test=df['Vanadium'].iloc[42:]
print(train,test)


# In[31]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[32]:


hwmodel = ExponentialSmoothing(train,trend='add', seasonal='mul', seasonal_periods=18).fit()


# In[33]:


pred = hwmodel.forecast(6)
pred


# In[34]:


train.plot(legend=True,label='train',figsize=(16,10))


# In[35]:


train.plot(legend=True,label='train',figsize=(16,10))
test.plot(legend=True,label='train',figsize=(16,10))


# In[36]:


train.plot(legend=True,label='train',figsize=(16,10))
test.plot(legend=True,label='test',figsize=(16,10))
pred.plot(legend=True,label='pred',figsize=(16,10))


# In[37]:


from sklearn.metrics import mean_squared_error


# In[38]:


mean_squared_error(test,pred)


# In[39]:


np.sqrt(mean_squared_error(test,pred))


# In[40]:


df['Vanadium'].mean(), np.sqrt(df['Vanadium'].var())


# In[41]:


#final model


# In[42]:


final_model =ExponentialSmoothing(Vanadium_series,trend='mul', seasonal='mul', seasonal_periods=6).fit()


# In[43]:


pred = final_model.forecast(6)
pred


# In[44]:


Vanadium_series.plot(legend=True, label='Vanadium',figsize=(16,10))

pred.plot(legend=True,label='Pred')


# # FB PROPHET

# In[40]:


from fbprophet import prophet


# In[41]:


get_ipython().system('pip install fbprophet')


# In[42]:


get_ipython().system('pip install fbprophet')


# In[43]:


get_ipython().system('pip from fbprophet import prophet')


# In[ ]:




