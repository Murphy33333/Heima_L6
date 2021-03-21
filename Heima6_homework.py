#coding by Murphy 2021/03/21
#ARMA模型
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm

train=pd.read_csv(r'D:\黑马课程\第六课\train.csv')
#转换成pandas日期格式
train['Datetime']=pd.to_datetime(train['Datetime'])

#将Datetime作为index
train.index=train['Datetime']
train.drop(['ID','Datetime'],axis=1,inplace=True)
print(train)
#安装天进行采样
daily_train=train.resample('D').sum()
print(daily_train)
# daily_train['ds']=daily_train.index
# daily_train['y']=daily_train['Count']
# daily_train.drop(['Count'],axis=1,inplace=True)
result=sm.tsa.stattools.adfuller(daily_train['Count'])
print(result)#不平稳差分，平稳不需要
#进行一次差分
diff1=daily_train['Count'].diff(1)
print(diff1[1:])
result1=sm.tsa.stattools.adfuller(diff1[1:])
print(result1)

#d=1差分一次
from statsmodels.tsa.arima_model import  ARIMA
model=ARIMA(daily_train,order=(7,0,5)).fit()
# arma = ARMA(daily_train,(7,0)).fit()
# print('AIC: %0.4lf' %arma.aic)

predict_y = model.predict('2014-12-10','2015-10-09',typ='levels')
print(predict_y)
plt.figure(figsize=(12,8))
predict_y.plot(color='r', ls='--', label='预测指数')
print(model.aic)
print('=================================split=========================================')
plt.show()



#fbprophet方法
import  pandas as pd
import matplotlib.pyplot as plt
#数据加载
train=pd.read_csv(r'D:\黑马课程\第六课\train.csv')

#转换成pandas日期格式
train['Datetime']=pd.to_datetime(train['Datetime'])

#将Datetime作为index
train.index=train['Datetime']
train.drop(['ID','Datetime'],axis=1,inplace=True)
print(train)
#安装天进行采样
daily_train=train.resample('D').sum()
print(daily_train)
daily_train['ds']=daily_train.index
daily_train['y']=daily_train['Count']
daily_train.drop(['Count'],axis=1,inplace=True)
print(daily_train)
from fbprophet import Prophet
#创建模型
m = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.1)
m.fit(daily_train)
# 预测未来7个月，213天
future = m.make_future_dataframe(periods=213)
print(future)
forecast=m.predict(future)
m.plot(forecast)
