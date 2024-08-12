import yfinance as yf
from yahoo_fin import stock_info
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
import numpy as np
idx = pd.IndexSlice

def strategy1(n):
    prices = pd.read_csv("data/prices.csv")
    count = pd.DataFrame(prices['Ticker'].value_counts()).reset_index()
    symbols = count[count['count'] >= 2520]['Ticker'].tolist()[:n]
    
    close = prices[prices['Ticker'].isin(symbols)][['Ticker', 'Close', 'Date']].set_index('Ticker')
    close = close.pivot_table(index='Date', columns='Ticker', values='Close')
    close.columns.name = None
    close.dropna(how="any", inplace= True)
    
    train_close, test_close = train_test_split(close, test_size=0.5, shuffle=False)
    corr_matrix = train_close.pct_change().corr(method ='pearson') #spearman
    corr_pairs = corr_matrix.unstack()
    corr_pairs = corr_pairs[corr_pairs != 1]
    selected_tickers = corr_pairs.sort_values(ascending=False).head(1)
    
    asset1 = selected_tickers.index[0][0]
    asset2 = selected_tickers.index[0][1]
    
    train = pd.DataFrame()
    train['asset1'] = train_close[asset1]
    train['asset2'] = train_close[asset2]
    
    model = sm.OLS(train.asset2, train.asset1).fit()
    
    # calculate z-score
    def zscore(series):
        return (series - series.mean()) / np.std(series)
    
    # create a dataframe for trading signals
    signals = pd.DataFrame()
    signals['asset1'] = test_close[asset1] 
    signals['asset2'] = test_close[asset2]
    ratios = signals.asset1 / signals.asset2
    
    # calculate z-score and define upper and lower thresholds
    signals['z'] = zscore(ratios)
    signals['z upper limit'] = np.mean(signals['z']) + np.std(signals['z'])
    signals['z lower limit'] = np.mean(signals['z']) - np.std(signals['z'])
    
    signals['signals1'] = 0
    signals['signals1'] = np.select([signals['z'] > signals['z upper limit'], signals['z'] < signals['z lower limit']], [-1, 1], default=0)
    
    # we take the first order difference to obtain portfolio position in that stock
    signals['positions1'] = signals['signals1'].diff()
    signals['signals2'] = -signals['signals1']
    signals['positions2'] = signals['signals2'].diff()


if __name__ == "__main__":
    data = pd.read_csv("data/nasdaq_screener.csv")
    #print(data['Industry'].value_counts() )
    symbols = data[data['Industry'] == 'Biotechnology: Pharmaceutical Preparations']['Symbol'].tolist()
    #print(tickers)
    price_list = []
    
    for symbol in symbols:
        try:
            price_temp = yf.download(symbol, period="max")
            price_temp['Ticker'] = symbol  # Add the ticker as a column
            price_list.append(price_temp)
        except:
            print(symbol)
            
    prices = pd.concat(price_list)
    #grouped = prices.groupby('Ticker')
    prices.to_csv("data/prices.csv")
            
    #industries = pd.DataFrame(data['Industry'].value_counts()).reset_index()
    #print(industries['count'].unique())
    #print(industries)
    #strategy1()
    