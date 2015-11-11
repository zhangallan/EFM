import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt

## Basics
s = pd.Series([1,2,3,4,5])

s = pd.Series([1,2,3,4,5], index = [chr(c) for c in range(ord('a'), ord('a') + 5)])

df = pd.DataFrame(dict(A = [1,2,3,4], B = np.random.randn(4)), index = ['a', 'b', 'c', 'd'])

## yahoo and csv handling
from pandas_datareader import data

aapl = pandas_datareader.data.get_data_yahoo('AAPL', 
                                             start=datetime.datetime(2010, 10, 1), 
                                             end=datetime.datetime(2012, 1, 1))

aapl.head()

msft = pandas_datareader.data.get_data_yahoo('MSFT', 
                                             start=datetime.datetime(2010, 10, 1), 
                                             end=datetime.datetime(2012, 1, 1))

msft.to_csv("msft_data.csv")
aapl.to_csv("aapl_data.csv")

msft.index

## slicing
s['a']

## fails!
df['a']

s[['a', 'd']]

s['a':'c':2]

# loc, iloc, ix
df.loc['a']

df.iloc[1]

df.ix[1, "A"]

aapl["Close"]

# Simple calculations
aapl["dailydiff"] = aapl["Close"] - aapl["Open"]

aapl.head()

del aapl["dailydiff"]

aapl.head()

## Boolean indexing
df['A'] > 3

df[df['A'] > 3]

df[(df['A'] > 2)]

df[~(df['A'] > 2)]

aapl[(aapl["Close"] > 300) & (aapl["Close"] < 500)].head()

# Useful functions
aapl.describe()

aapl.sort_index(axis = 1).head()

aapl.sort_values(by = "Volume", ascending = False).head()

aapl.apply(np.mean)

# Moving average
aapl_mavg = pd.rolling_mean(aapl["Close"], 30)

aapl_mavg.name = "Moving Average"
aapl_mavg.head()
aapl_mavg.tail()

# Missing values
aapl_mavg.head()
aapl_mavg.isnull()

aapl_mavg[~aapl_mavg.isnull()]

aapl_mavg.fillna(value = 5)

aapl_mavg.dropna(how = "any")

# Basic plotting
aapl["Close"].plot()
plt.show()

aapl["Open"].plot()
aapl["Close"].plot()
aapl_mavg.plot()
plt.legend()
plt.show()

aapl["Return"] = (aapl["Close"] - aapl["Close"].shift(1)) / aapl["Close"].shift(1)

aapl["Return"].head()

aapl["Return"].plot()
plt.show()

# Time series stuff
aapl.index

aapl.index[0]

weekly = pd.date_range("2010-10-01", "2011-12-30", freq = 'W-MON')
weekly

aapl["Return"][weekly]

aapl.index.week

aapl.index.month

aapl["Return"][aapl.index.month == 10]

# Groupby

aapl.groupby(aapl.index.month)
aapl.groupby(aapl.index.month).describe()
aapl.groupby(aapl.index.month).plot()
plt.show()


# Fun stuff
stock_data = pandas_datareader.data.get_data_yahoo(['AAPL', 'IBM', 'MSFT', 'XOM', 'MON'], 
                               start=datetime.datetime(2010, 1, 1), 
                               end=datetime.datetime(2013, 1, 1))

stock_data

stock_data["Adj Close"].to_csv("stock_data_adjclose.csv")

returns = stock_data["Adj Close"].pct_change()

# Scatter plot
plt.scatter(returns["MSFT"], returns["AAPL"])
plt.plot([returns["MSFT"].min(),returns["MSFT"].max()],[returns["MSFT"].min(),returns["MSFT"].max()])
# plt.plot([-.1,.1], [-.1,.1])
plt.show()

# Returns correlation
returns.corr()

# Scatter matrix
pd.scatter_matrix(returns)
plt.show()

# heatmap
plt.imshow(returns.corr(), cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(returns.corr())), returns.columns)
plt.yticks(range(len(returns.corr())), returns.columns)
plt.show()

#### Bad trading strategy
# trading = aapl[["Close", "Open"]].copy()

# trading.loc[:, "Order"] = 0

# trading["Return"] = trading["Close"].pct_change()

# trading["Week Moving Avg"] = pd.rolling_mean(trading["Close"], 7)

# trading.loc[trading["Week Moving Avg"] < trading["Close"], "Order"] = 1
# trading.loc[trading["Week Moving Avg"] >= trading["Close"], "Order"] = -1

# trading["Strat Return"] = trading.Order * trading.Return

# (1 + trading["Strat Return"]).cumprod().plot();
# plt.ylabel('Portfolio value')
# plt.show()

