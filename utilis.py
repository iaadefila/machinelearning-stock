
from sklearn.preprocessing import StandardScaler
import datetime as dt
import numpy as np
import pandas as pd 
from finta import TA
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


def LSTM_pred(df, model):
    series = df[["Close",	"MOM",	"RSI",	"ROC",	"MACD",	"BB",	"CHAIKIN",	"STOCH"]]
    
    sc_X = StandardScaler()
    series.iloc[:,1:] = sc_X.fit_transform(series.iloc[:,1:])
    sc_y = StandardScaler()
    series[['Close']] = sc_y.fit_transform(series[['Close']])
        
    data = series.values
    timesteps = 45
    X_data = []
    Y_data = []
    
    # Loop for testing data
    for i in range(timesteps,data.shape[0]):
        X_data.append(data[i-timesteps:i])
        Y_data.append(data[i][0])
    X_data,Y_data = np.array(X_data),np.array(Y_data)
    
    last_days = np.array([X_data[-1]])
    Y_hat = model.predict(last_days)
    Y_pred = sc_y.inverse_transform(Y_hat.T)
    Y_pred = Y_pred.tolist()
    pred = [round(x[0], 2) for x in Y_pred]
    
    
    return pred

def SARIMAX_pred(df, model):
    series = df[["Close",	"MOM",	"RSI",	"ROC",	"MACD",	"BB",	"CHAIKIN",	"STOCH"]] # Picking the series with high correlation
    steps=-1
    dataset_with_step= series.copy()
    dataset_with_step['Actual']=dataset_with_step['Close'].shift(steps)
    dataset_with_step.dropna(inplace=True)
    # normalizing input features
    sc_in = StandardScaler()
    scaled_input = sc_in.fit_transform(dataset_with_step[["Close",	"MOM",	"RSI",	"ROC",	"MACD",	"BB",	"CHAIKIN",	"STOCH"]])
    scaled_input =pd.DataFrame(scaled_input)
    X= scaled_input
    
    # normalizing output features
    sc_out = StandardScaler()
    scaler_output = sc_out.fit_transform(dataset_with_step[['Actual']])
    scaler_output =pd.DataFrame(scaler_output)
    y=scaler_output
    
    #Ceeate a dataframe to work with it after scaling 
    X.rename(columns={0:'Close', 1:'MOM', 2:'RSI', 3:'ROC', 4:'MACD', 5:'BB', 6:'CHAIKIN',7:"STOCH"}, inplace=True)
    X= pd.DataFrame(X)
    X.index=dataset_with_step.index
    y.rename(columns={0:'Stock Price next day'}, inplace= True)
    y.index=dataset_with_step.index
    
    X_test = X.iloc[[-1]]

    forecast = model.forecast(steps=1, exog=X_test)
    y_pred_inv = sc_out.inverse_transform(np.array([forecast]))
    
    pred = round(y_pred_inv.tolist()[0][0],2)
    
    return pred
    
def ConvLstm_pred(df, model):
    series = df[["Close",	"MOM",	"RSI",	"ROC",	"MACD",	"BB",	"CHAIKIN",	"STOCH"]] # Picking the series with high correlation
    
    sc_X = StandardScaler()
    series.iloc[:,1:] = sc_X.fit_transform(series.iloc[:,1:])
    sc_y = StandardScaler()
    series[['Close']] = sc_y.fit_transform(series[['Close']])
        
    data = series.values
    timesteps = 45
    X_data = []
    Y_data = []
    
    # Loop for testing data
    for i in range(timesteps,data.shape[0]):
        X_data.append(data[i-timesteps:i])
        Y_data.append(data[i][0])
    X_data,Y_data = np.array(X_data),np.array(Y_data)
    X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1, X_data.shape[2], 1)
    
    last_days = np.array([X_data[-1]])
    Y_hat = model.predict(last_days)
    Y_pred = sc_y.inverse_transform(Y_hat.T)
    Y_pred = Y_pred.tolist()
    pred = [round(x[0], 2) for x in Y_pred]

    return pred 
    
    
    
    
def removecomma(x):
    if isinstance(x, str):
        return(x.replace(',', ''))
    return(x)


def preprocess(dataset_url):
    df = pd.read_csv(dataset_url,parse_dates = True,index_col=0)
    # Reformat the Close, Open, High and Low features
    df['close'] = df['close'].apply(removecomma).astype('float')
    df['open'] = df['open'].apply(removecomma).astype('float')
    df['high'] = df['high'].apply(removecomma).astype('float')
    df['low'] = df['low'].apply(removecomma).astype('float')

    # Reformat the Volume feature
    df.volume = df.volume.replace('-', '0')
    df.volume = (df.volume.replace(r'[KMB]+$', '', regex=True).astype(float) * df.volume.str.extract(r'[\d\.]+([KMB]+)', expand=False).fillna(1).replace(['K','M','B'], [10**3, 10**6, 10**9]).astype(int))
    # Create other features
    df ['MOM'] = TA.MOM(df)
    df ['RSI'] = TA.RSI(df, 21)
    df ['ROC'] = TA.ROC(df, 21)
    df ['MACD'] = TA.MACD(df)['MACD']
    df ['BB'] = TA.BBANDS(df)['BB_MIDDLE']
    df ['CHAIKIN'] = TA.CHAIKIN(df)
    df ['STOCH'] = TA.STOCH(df)
    
    # Remove Null values
    df.dropna(inplace=True)
    # Remove the Symbol, Open, High, Low and Volume features 
    df.drop(columns={'open','high','low','volume'}, inplace = True)
    # Rename the feature names
    df.rename(columns = {"close":"Close"}, inplace = True)
    
    
    return df 

def cp(dataset_url):
    cdf = pd.read_csv(dataset_url,parse_dates = True,index_col=0)
    current = cdf.iloc[-1]['close'].astype('float')
    return current  
        
    

