#!/usr/bin/env python
# coding: utf-8

# 
# # Sharps ratio
# <p>An investment may make sense if we expect it to return more money than it costs. But every investment has an intrinsic risk. We can invest our capital in a safer way with smaller profit gains. Nevertheless, some risks might be worth it and if we have the data we can tell when are worth it.</p>
# <p>When faced with investment alternatives that offer both different returns and risks, the Sharpe Ratio helps to make a decision by adjusting the returns by the differences in risk and allows an investor to compare investment opportunities on equal terms, that is, on an 'apples-to-apples' basis.</p>
# <p>Let's apply the Shape Ratio by calculating it for the stocks of Facebook and Amazon. Our benchmark database will be the S&amp;P 500, which measures the performance of the 500 largest stocks in the US.</p>
# 
# *The Sharps ratio is named that way in honor of Professor William Sharpe, nobel price in economics and creator of the Sharps ratio method.

# In[1]:


# Importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Settings to produce plots with the style color of Fivethirtyeight
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

# Reading in the data
stock_data = pd.read_csv("datasets/stock_data.csv", 
    parse_dates=['Date'],
    index_col=['Date']
    ).dropna()
benchmark_data = pd.read_csv("datasets/benchmark_data.csv", 
    parse_dates=['Date'],
    index_col=['Date']
    ).dropna()


# ## 2. A first glance at the data
# <p>Let's take a look the data to find out how many observations and variables we have at our disposal.</p>
# <p>First, we show the topr fows of the database followed by general information of the dataframe.</p>
# 
# ### Stock Data

# In[2]:


print("\nTable Headers")
print(stock_data.head())
print("\nInfo")
stock_data.info()


# ### Benchmark

# In[3]:


print("\nTable Headers")
print(benchmark_data.head())
print("\nInfo")
benchmark_data.info()


# ## 3. Plot & summarize daily prices for Amazon and Facebook
# <p>Before we compare an investment in either Facebook or Amazon with the index of the 500 largest companies in the US, let's visualize the data, so we better understand what we need to handle.</p>
# 
# <p>Place particular attention on the scale of each subplot.</p>

# In[4]:


# visualizes the stock_data
stock_data.plot(title='some title', subplots=True)

# summarizes the stock_data
stock_data.describe()


# ## 4. Visualize & summarize daily values for the S&P 500
# <p>Let's also take a closer look at the value of the S&amp;P 500, our benchmark.</p>

# In[5]:


# plots the benchmark_data
benchmark_data.plot(title = "S&P 500")


# summarizes the benchmark_data
benchmark_data.describe()


# ## 5. The inputs for the Sharpe Ratio: Starting with Daily Stock Returns
# <p>The Sharpe Ratio uses the difference in returns between the two investment opportunities under consideration.</p>
# <p>However, our data show the historical value of each investment, not the return. To calculate the return, we need to calculate the percentage change in value from one day to the next. We'll also take a look at the summary statistics because these will become our inputs as we calculate the Sharpe Ratio. Can you already guess the result?</p>

# In[6]:


stock_returns = stock_data.pct_change()
stock_returns.plot();
stock_returns.describe()


# ## 6. Daily S&P 500 returns
# <p>For the S&amp;P 500, calculating daily returns works just the same way, we just need to make sure we select it as a <code>Series</code> using single brackets <code>[]</code> and not as a <code>DataFrame</code> to facilitate the calculations in the next step.</p>

# In[7]:


# daily benchmark_data returns
sp_returns = benchmark_data["S&P 500"].pct_change()

# plots the daily returns
sp_returns.plot()

# summary of the daily returns
sp_returns.describe()


# ## 7. Calculating Excess Returns for Amazon and Facebook vs. S&P 500
# <p>Next, we need to calculate the relative performance of stocks vs. the S&amp;P 500 benchmark. This is calculated as the difference in returns between <code>stock_returns</code> and <code>sp_returns</code> for each day. This is what tells us how much could we do after each exchange.</p>

# In[8]:


# calculates the difference in daily returns
excess_returns = stock_returns.sub(sp_returns, axis=0)

# plots of the excess_returns
excess_returns.plot()

# summarizes the excess_returns
excess_returns.describe()


# ## 8. The Sharpe Ratio, Step 1: The Average Difference in Daily Returns Stocks vs S&P 500
# <p>Now we can finally start computing the Sharpe Ratio. First we need to calculate the average of the <code>excess_returns</code>. This tells us how much more or less the investment yields per day compared to the benchmark.</p>

# In[9]:


# Calculates the mean of the excess returns 
avg_excess_return = excess_returns.mean()

# Plots the averages
avg_excess_return.plot.bar(title ="Mean of the return difference")


# ## 9. The Sharpe Ratio, Step 2: Standard Deviation of the Return Difference
# <p>The stock price changes every time a stock is traded, and that change is not always for good. There is where the risk of trading lies. We can measure that using the standard deviation from the mean as proposed by the Shapes ratio </p>
#     
# <p>It looks like there was quite a bit of a difference between average daily returns for Amazon and Facebook.</p>
# <p>Next, we calculate the standard deviation of the <code>excess_returns</code>. This shows us the amount of risk investment in the stocks implies as compared to an investment in the S&amp;P 500.</p>

# In[10]:


# calculates the standard deviations
sd_excess_return = excess_returns.std()

# plots the standard deviations
sd_excess_return.plot.bar(title = 'Standard Deviation of the Return Difference')


# ## 10. Putting it all together
# <p>Now we just need to compute the ratio of <code>avg_excess_returns</code> and <code>sd_excess_returns</code>. The result is now finally the <em>Sharpe ratio</em> and indicates how much more (or less) return the investment opportunity under consideration yields per unit of risk.</p>
# <p>The Sharpe Ratio is often <em>annualized</em> by multiplying it by the square root of the number of periods. We have used daily data as input, so we'll use the square root of the number of trading days (5 days, 52 weeks, minus a few holidays): √252</p>

# In[11]:


# calculates the daily sharpe ratio
daily_sharpe_ratio = avg_excess_return.div(sd_excess_return )

# annualizes the sharpe ratio
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)

# plots the annualized sharpe ratio
annual_sharpe_ratio.plot.bar(title="Annualized Sharpe Ratio: Stocks vs S&P 500")


# ## 11. Conclusion
# <p>Given the two Sharpe ratios, which investment should we go for? In 2016, Amazon had a Sharpe ratio twice as high as Facebook. This means that an investment in Amazon returned twice as much compared to the S&amp;P 500 for each unit of risk an investor would have assumed. In other words, in risk-adjusted terms, the investment in Amazon would have been more attractive.</p>
# <p>This difference was mostly driven by differences in return rather than risk between Amazon and Facebook. The risk of choosing Amazon over FB (as measured by the standard deviation) was only slightly higher so that the higher Sharpe ratio for Amazon ends up higher mainly due to the higher average daily returns for Amazon. </p>
# 
# <p>Now is time to put your money work for you, and a tool to find the best deal.. We can extrapolate this to find the best deal between multiple companies using a similar approach</p>

# In[ ]:




