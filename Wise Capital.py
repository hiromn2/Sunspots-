#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 23:54:10 2025

@author: hiro
"""


import os 
import numpy as np
import yfinance as yf
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats



print(os.getcwd())
new_directory_path = "/Users/hiro/documents/github/Wise Capital"
os.chdir(new_directory_path)
print(os.getcwd())




cols = ['Year', 'Month', 'Date', 'Sunspot', 'SD', 'N_Obs', 'Provision']
df = pd.read_csv('SN_m_tot_V2.0.csv', sep=';', header=None, names=cols)
df.columns

df = df.replace(-1, np.nan)
df = df[df['Year'] >= 2005]
df = df.astype(float, errors="ignore")

# Use an explicit end date for reproducibility (set to "today" when you run)
start = "2005-01-01"  # inclusive
end   = "2025-12-01"  # exclusive (use first day of next month)

ticker = "^GSPC"  # S&P 500 index on Yahoo Finance :contentReference[oaicite:1]{index=1}

sp = yf.download(
    ticker,
    start=start,
    end=end,
    interval="1mo",
    auto_adjust=False,  # keep raw OHLC; set True if you want adjusted prices
    actions=False,      # dividends/splits are not relevant for an index, but keep it explicit
    progress=False,
)


sp.index.name = "Date"

# Optional: enforce column names and types
#sp.columns = [c.lower().replace(" ", "_") for c in df.columns]
sp = sp.astype(float, errors="ignore")
sp = sp['Adj Close']
sp.columns = ['sp']
sp = np.log(sp / sp.shift(1))
sp.to_parquet("sp500_gspc_daily_last20y.parquet")
sp = sp.dropna()


sns.histplot(sp, bins = 30, color = 'skyblue', edgecolor = 'black')
plt.show()
sns.kdeplot(sp, color = 'skyblue')
plt.show()

print(sp.describe())
print(f"Skewness: {sp.mean()}")  # 0.00703
print(f"Skewness: {sp.var()}")  # 0.001873
print(f"Skewness: {sp.skew()}")  # -0.814331
print(f"Kurtosis: {sp.kurtosis()}")  #1.766

# Jarque-Bera test 
jb_stat, jb_pvalue = stats.jarque_bera(sp.dropna())
print(f"Jarque-Bera test:")
print(f"Statistic: {jb_stat:.4f}, p-value: {jb_pvalue:.4f}")
print(f"Reject normality: {jb_pvalue < 0.05}")

# Shapiro-Wilk test 
sw_stat, sw_pvalue = stats.shapiro(sp.dropna())
print(f"\nShapiro-Wilk test:")
print(f"Statistic: {sw_stat:.4f}, p-value: {sw_pvalue:.4f}")
print(f"Reject normality: {sw_pvalue < 0.05}")


df['sp'] = sp


import matplotlib.pyplot as plt

sp = sp.squeeze()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram with normal overlay

axes[0, 0].hist(sp, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
mu, sigma = sp.mean(), sp.std()
x = np.linspace(sp.min(), sp.max(), 100)
axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
axes[0, 0].set_title('Histogram vs Normal Distribution')
axes[0, 0].legend()
axes[0, 0].set_xlabel('Monthly Log Returns')



# 2. Q-Q Plot
stats.probplot(sp.squeeze(), dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot (Normal)')


# 3. KDE plot
sns.kdeplot(sp, ax=axes[1, 0], color='skyblue', fill=True)
axes[1, 0].axvline(mu, color='r', linestyle='--', label=f'Mean: {mu:.4f}')
axes[1, 0].axvline(0, color='k', linestyle='-', alpha=0.3)
axes[1, 0].set_title('Kernel Density Estimate')
axes[1, 0].legend()

# 4. Time series plot
sp.plot(ax=axes[1, 1], color='skyblue')
axes[1, 1].set_title('Time Series of Returns')
axes[1, 1].axhline(0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()


# Student's t-distribution (common for financial returns)
params_t = stats.t.fit(sp)
df_t, loc_t, scale_t = params_t
print(f"\nStudent's t-distribution:")
print(f"Degrees of freedom: {df_t:.2f}")  # Lower = fatter tails

# Compare fits
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(sp, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')

x = np.linspace(sp.min(), sp.max(), 100)
ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
ax.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t), 'g-', linewidth=2, label="Student's t")
ax.legend()
ax.set_title('Distribution Fits')
plt.show()

# Kolmogorov-Smirnov test
ks_norm = stats.kstest(sp, 'norm', args=(mu, sigma))
ks_t = stats.kstest(sp, 't', args=params_t)
print(f"\nKS test (Normal): p-value = {ks_norm.pvalue:.4f}")
print(f"KS test (t-dist): p-value = {ks_t.pvalue:.4f}")

# Identify extreme events (>3 standard deviations)
threshold = 3
extreme_events = sp[np.abs(sp - mu) > threshold * sigma]
print(f"\nExtreme events (>{threshold}σ): {len(extreme_events)}")
print(extreme_events.sort_values())

# Expected under normal distribution
expected_extreme = len(sp) * (1 - stats.norm.cdf(threshold) + stats.norm.cdf(-threshold))
print(f"Expected under normal: {expected_extreme:.1f}")
print(f"Actual: {len(extreme_events)} (excess: {len(extreme_events) - expected_extreme:.1f})")




