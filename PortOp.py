#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# In[2]:


# Step 2: Data Retrieval

# Define the list of assets
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 'WMT', 'PG']

# Define the time period
start_date = '2018-01-01'
end_date = '2023-10-01'

# Fetch the adjusted closing prices
data = yf.download(assets, start=start_date, end=end_date)['Adj Close']
data.head()


# In[4]:


# Step 3: Data Preprocessing

# Check for missing values
print(data.isnull().sum())

# Forward fill to handle missing data
data.ffill(inplace=True)

# Calculate daily returns
returns = data.pct_change().dropna()
returns.head()


# In[5]:


# Step 4: Calculating Expected Returns and Covariance Matrix

# Calculate annualized expected returns
expected_returns = returns.mean() * 252

# Calculate annualized covariance matrix
cov_matrix = returns.cov() * 252


# In[6]:


# Step 5: Portfolio Optimization using Mean-Variance Optimization

# Number of assets
num_assets = len(assets)

# Define the optimization variables
weights = cp.Variable(num_assets)

# Define the expected portfolio return
portfolio_return = expected_returns.values @ weights

# Define the portfolio risk (variance)
portfolio_risk = cp.quad_form(weights, cov_matrix.values)

# Define the objective function (minimize risk)
objective = cp.Minimize(portfolio_risk)

# Constraints
constraints = [
    cp.sum(weights) == 1,        # Weights sum to 1
    weights >= 0,                # No short-selling
    portfolio_return >= 0.20     # Target return of 20%
]

# Define the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

# Optimal weights
optimal_weights = weights.value


# In[7]:


# Step 6: Incorporating Constraints and Extensions

# Let's add a maximum weight constraint per asset
max_weight = 0.20
constraints.append(weights <= max_weight)

# Re-define and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Updated optimal weights
optimal_weights = weights.value


# In[8]:


# Step 7: Optimization with CVXPY

# Re-run the optimization with the new constraints
problem.solve(solver=cp.SCS)

# Check if the problem is solved
print(f"Problem Status: {problem.status}")

# Optimal portfolio performance
optimal_return = expected_returns.values @ optimal_weights
optimal_risk = np.sqrt(optimal_weights.T @ cov_matrix.values @ optimal_weights)

print(f"Optimal Expected Return: {optimal_return:.2%}")
print(f"Optimal Portfolio Risk: {optimal_risk:.2%}")


# In[9]:


# Step 8: Analyzing and Visualizing the Results

# Create a DataFrame for the optimal weights
portfolio = pd.DataFrame({'Asset': assets, 'Weight': optimal_weights})

# Plot the asset allocation
plt.figure(figsize=(10, 6))
sns.barplot(x='Asset', y='Weight', data=portfolio)
plt.title('Optimized Portfolio Allocation')
plt.ylabel('Weight')
plt.xlabel('Asset')
plt.show()


# In[10]:


# Step 9: Backtesting the Optimized Portfolio

# Calculate the portfolio returns
portfolio_returns = returns @ optimal_weights

# Calculate cumulative returns
cumulative_returns = (1 + portfolio_returns).cumprod()

# Plot the cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns, label='Optimized Portfolio')
plt.title('Portfolio Backtest Results')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()


# In[11]:


# Step 10: Comparing with Equal Weight Portfolio

# Equal weights
equal_weights = np.array([1/num_assets]*num_assets)

# Equal weight portfolio returns
equal_portfolio_returns = returns @ equal_weights

# Equal portfolio cumulative returns
equal_cumulative_returns = (1 + equal_portfolio_returns).cumprod()

# Plotting both portfolios
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns, label='Optimized Portfolio')
plt.plot(equal_cumulative_returns, label='Equal Weight Portfolio')
plt.title('Optimized vs. Equal Weight Portfolio')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()


# In[ ]:




