# Importing
import pandas as pd
import numpy as np

import datetime

import matplotlib.pyplot as plt

# Series and DataFrame creation
s = pd.Series([1,2,3,4])

# Include index
s = pd.Series([1,2,3,4], index = ['a', 'b', 'c', 'd'])

# DataFrame creation with dictionary
df = pd.DataFrame(dict(A = [1,2,3,4], B = [10,12,13,14]))

# yahoo data
from pandas_datareader import data

# Get Apple and Microsoft data from Yahoo
aapl = data.get_data_yahoo('AAPL', 
                           start=datetime.datetime(2010, 10, 1), 
                           end=datetime.datetime(2012, 1, 1))

aapl.head()

msft = data.get_data_yahoo('MSFT', 
                           start=datetime.datetime(2010, 10, 1), 
                           end=datetime.datetime(2012, 1, 1))

# Getting rows and columns from Series and DataFrame example
s['a']

# Fails! First slice in dataframes is the column. Use the loc function (below) if this confuses you
df['a']

# Multiple rows
s[['a', 'd']]

# List indexing and syntax works here too!
s['a':'c':2]

# DataFrame selecting is reversed
df["A"]
df["A"][1]

aapl["Close"]

# loc (index based selection), iloc (integer based selection), ix (select through both ways)
df.iloc[2,0]

# Equivalent and grab the same value
s.loc['c']
s.iloc[2]

# Simple calculations and assignment
aapl["Close Minus Open"] = aapl["Close"] - aapl["Open"]

aapl.head()

# Delete column
del aapl["Close Minus Open"]

aapl.head()

# Boolean indexing
df[df["B"] > 10]

aapl.head()

aapl[aapl["Close"] > 400]

# ~ is the not syntax for boolean indexing. You can also use & and | for and and or
aapl[(aapl["Close"] < 300) & ~(aapl["Close"] > 400)]

# Useful functions
df.mean()

aapl.describe()

# Sorts the columns by name
aapl.sort_index(axis = 1).head()

# Sort the dataframe by the values in a single column
aapl.sort_values(by = "Volume").head()
aapl.sort_values(by = "Volume", ascending = False).head()

# Apply a function across columns
aapl.apply(np.mean)

# More advanced example. Centers all columns of the data frame. Very useful for statistical analysis
def standardize(x):
    return (x - np.mean(x)) / np.std(x)

aapl.apply(standardize)

# Same as above with a lambda function
aapl.apply(lambda x: (x - np.mean(x)) / np.std(x))

# Moving average
aapl_mavg = pd.rolling_mean(aapl["Close"], 30)

# Missing Values
aapl_mavg.head()
aapl_mavg.isnull()

aapl_mavg[~(aapl_mavg.isnull())]

# Drop all rows where ALL of the values are NA
aapl_mavg.dropna(how = "all")

aapl_mavg.fillna(value = 5)

# Fill it with the close values. Needs to have the same index value at the points to be filled
aapl_mavg.fillna(value = aapl["Close"])

# Basic Plotting. Multiple plots can be plotted by just calling plot multiple times before plt.show()
aapl["Close"].plot()
aapl_mavg.name = "Moving Average"
aapl_mavg.plot()
plt.legend()
plt.show()

# Also has multiple types of plots. Check documentation for more.
aapl["Close"].plot(kind = "hist")
plt.show()

# Time series stuff
aapl.index[0]

# Date range of only Mondays in the time period
weekly = pd.date_range("2010-10-01", "2011-12-30", freq = "W-MON")

aapl["Close"][weekly]

# Allows you to get the week number and month number for every index
aapl.index.week

aapl.index.month

# Closes for only October
aapl["Close"][aapl.index.month == 10]

# Groupby month
aapl.groupby(aapl.index.month).describe()

aapl.groupby(aapl.index.month).plot()
plt.show()

# Fun stuff

# Returns a panel frame. Not too different from dataframe other than you can't look into it
stock_data = data.get_data_yahoo(['AAPL', 'IBM', 'MSFT', 'XOM', 'MON'], 
                               start=datetime.datetime(2010, 1, 1), 
                               end=datetime.datetime(2013, 1, 1))

# Grabbing a dataframe out of a panel
stock_data["Adj Close"]

returns = stock_data["Adj Close"].pct_change()

# Scatter plot. Matplotlib function
plt.scatter(returns["MSFT"], returns["AAPL"])

# A bit of code to just draw line with slope 1. I don't know why it's so hard to do this in matplotlib...
plt.plot([returns["MSFT"].min(),returns["MSFT"].max()],[returns["MSFT"].min(),returns["MSFT"].max()])
plt.show()

# Correlation matrix for your data
returns.corr()

# Scatter plot matrix
pd.scatter_matrix(returns)
plt.show()

# Heatmap. Matplotlib function
plt.imshow(returns.corr(), cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(returns.corr())), returns.columns)
plt.yticks(range(len(returns.corr())), returns.columns)
plt.show()
