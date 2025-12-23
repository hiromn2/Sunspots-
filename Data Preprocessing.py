"""
Data Preprocessing, handling and cleaning.
Additionally, I perform some statistical checks to see if the downloaded data is reasonable.

1. Read Sunspot data
2. Download S&P 500 data
3. Preprocess
4. Statistical EDA 
5. Download controls
6. Format and export

"""

import os 
import numpy as np
import yfinance as yf
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from pandas_datareader import data as pdr

# Setting the directory
print(os.getcwd())
new_directory_path = "/Users/hiro/documents/github/Wise Capital" #Change to your directory
os.chdir(new_directory_path)
print(os.getcwd())


cols = ['Year', 'Month', 'Date', 'Sunspot', 'SD', 'N_Obs', 'Provision']
df = pd.read_csv('SN_m_tot_V2.0.csv', sep=';', header=None, names=cols)

df = df.replace(-1, np.nan) #All the NAN are marked as -1 in the data
df = df[df['Year'] >= 2005].copy() #Let's keep just the last 20 years

sun_dates = pd.to_datetime(
    dict(year=df['Year'].astype(int), month=df['Month'].astype(int), day=1)
) + pd.offsets.MonthEnd(0)

sun = (
    df.assign(Date=sun_dates)
      .set_index('Date')['Sunspot']
      .astype(float)
      .sort_index()
      .dropna()
)

start = "2005-01-01"  
end   = "2025-12-01"  


sp_raw = yf.download(
    "^GSPC",
    start=start,
    end=end,
    interval="1mo",
    auto_adjust=False,
    actions=False,
    progress=False,
)



# 1) Remove the ticker level from columns (if present)
if isinstance(sp_raw.columns, pd.MultiIndex):
    sp_raw.columns = sp_raw.columns.droplevel(1)

sp_px = sp_raw["Adj Close"].copy()
sp_px.index = sp_px.index.to_period("M").to_timestamp("M")

sp = np.log(sp_px / sp_px.shift(1)).rename("returns").dropna()

# now this will align
data = pd.concat([sp, sun.rename("sunspot")], axis=1, join="inner").dropna()


sp.to_csv("data/sp500.csv")

# =========================================================================================================
# Basic Statistics
# =========================================================================================================


print(sp.describe())
print(f"Mean: {sp.mean()}")  # 0.00703
print(f"Variance: {sp.var()}")  # 0.001873
print(f"Skewness: {sp.skew()}")  # -0.814331
print(f"Kurtosis: {sp.kurtosis()}")  #1.766


print(df['Sunspot'].describe())
print(f"Mean: {df['Sunspot'].mean()}")  #54.76454183266933
print(f"Variance: {df['Sunspot'].var()}")  #2394.684457689243
print(f"Skewness: {df['Sunspot'].skew()}")  #0.7423414873912438
print(f"Kurtosis: {df['Sunspot'].kurtosis()}")  #-0.39692713725637185


# =========================================================================================================
# Normality Tests
# =========================================================================================================


# Log Returns
# Jarque-Bera test 
jb_stat, jb_pvalue = stats.jarque_bera(sp.dropna())
print(f"Jarque-Bera test:")
print(f"Statistic: {jb_stat:.4f}, p-value: {jb_pvalue:.4f}")
print(f"Reject normality: {jb_pvalue < 0.05}") #Normality rejected

# Shapiro-Wilk test 
sw_stat, sw_pvalue = stats.shapiro(sp.dropna())
print(f"\nShapiro-Wilk test:")
print(f"Statistic: {sw_stat:.4f}, p-value: {sw_pvalue:.4f}")
print(f"Reject normality: {sw_pvalue < 0.05}") #Normality rejected

# As expected, the log returns are not normal

# Sunspots
# Jarque-Bera test 
jb_stat_sunspot, jb_pvalue_sunspot = stats.jarque_bera(df['Sunspot'].dropna())
print(f"Jarque-Bera test:")
print(f"Statistic: {jb_stat_sunspot:.4f}, p-value: {jb_pvalue_sunspot:.4f}")
print(f"Reject normality: {jb_pvalue_sunspot < 0.05}") #Normality rejected 

# Shapiro-Wilk test 
sw_stat_sunspot, sw_pvalue_sunspot = stats.shapiro(df['Sunspot'].dropna())
print(f"\nShapiro-Wilk test:")
print(f"Statistic: {sw_stat_sunspot:.4f}, p-value: {sw_pvalue_sunspot:.4f}")
print(f"Reject normality: {sw_pvalue_sunspot < 0.05}") #Normality rejected



# =========================================================================================================
# Exploratory Data Analysis
# =========================================================================================================

# Log Returns
sns.set_style('ticks') 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram
axes[0, 0].hist(sp, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
mu, sigma = float(sp.mean()), float(sp.std())
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
sns.set_style('whitegrid')
OUTPUT_DIR = 'Plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(f'{OUTPUT_DIR}/distribution_analysis_log_returns.png', dpi=300, bbox_inches='tight')
plt.show()





# Sunspots

sns.set_style('ticks') 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram
axes[0, 0].hist(df['Sunspot'], bins=30, density=True, alpha=0.7, color='orange', edgecolor='black')
mu, sigma = float(df['Sunspot'].mean()), float(df['Sunspot'].std())
x = np.linspace(df['Sunspot'].min(), df['Sunspot'].max(), 100)
axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
axes[0, 0].set_title('Histogram vs Normal Distribution')
axes[0, 0].legend()
axes[0, 0].set_xlabel('Monthly Log Returns')


# 2. Q-Q Plot
stats.probplot(df['Sunspot'].squeeze(), dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot (Normal)')


# 3. KDE plot
sns.kdeplot(df['Sunspot'], ax=axes[1, 0], color='orange', fill=True)
axes[1, 0].axvline(mu, color='r', linestyle='--', label=f'Mean: {mu:.4f}')
axes[1, 0].axvline(0, color='k', linestyle='-', alpha=0.3)
axes[1, 0].set_title('Kernel Density Estimate')
axes[1, 0].legend()

# 4. Time series plot
df['Sunspot'].plot(ax=axes[1, 1], color='orange')
axes[1, 1].set_title('Time Series of Returns')
axes[1, 1].axhline(0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
sns.set_style('whitegrid')
OUTPUT_DIR = 'Plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.savefig(f'{OUTPUT_DIR}/distribution_analysis_sunspot.png', dpi=300, bbox_inches='tight')
plt.show()




# =========================================================================================================
# Adding controls
# =========================================================================================================
fred_series = [
    "TB3MS",      # 3-Month Treasury Bill rate
    "T10Y3MM",    # 10Y-3M Treasury spread (yield curve)
    "AAA",        # AAA corporate bond yield
    "BAA",        # BAA corporate bond yield
    "CPIAUCSL",   # Consumer Price Index
    "INDPRO",     # Industrial Production Index
    "UNRATE",     # Unemployment Rate
    "UMCSENT",    # Consumer Sentiment
    "VIXCLS"      # VIX volatility index
]



controls = pdr.DataReader(fred_series, "fred", "2005-01-01", "2025-12-01")
    
controls = pd.DataFrame({
    "tb3m": controls["TB3MS"].resample("ME").last(),
    "yield_spread": controls["T10Y3MM"].resample("ME").last(),
    "aaa_yield": controls["AAA"].resample("ME").last(),
    "baa_yield": controls["BAA"].resample("ME").last(),
    "inflation_yoy": controls["CPIAUCSL"].pct_change(12).resample("ME").last(),  # Year-over-year
    "ip_growth": controls["INDPRO"].pct_change(1).resample("ME").last(),  # Monthly growth
    "unemployment": controls["UNRATE"].resample("ME").last(),
    "consumer_sent": controls["UMCSENT"].resample("ME").last(),
    "vix": controls["VIXCLS"].resample("ME").mean()  # Average VIX for the month
})

# Create credit spread (common risk indicator)
controls["credit_spread"] = controls["baa_yield"] - controls["aaa_yield"]

n = min(len(sp), len(df['Sunspot']))

data = pd.concat(
    [sp, sun],
    axis=1,
    join='inner'
)

controls_lagged = controls.shift(1)
data = data.join(controls_lagged, how='inner')

print(f"Variables: {list(data.columns)}")
data.to_csv("data/Wise_Capital_Data.csv")

df.to_csv("df.csv")


