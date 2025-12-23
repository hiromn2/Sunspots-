"""
Data Analysis. We are looking for evidence (or discredit) on whether sunspots affect the returns on the stock market.

1. Read the clean data
2. Test stationarity 
3. Linear regression analysis
4. Adding some more controls

"""

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_breuschpagan,
    het_white,
    linear_reset
)
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas_datareader import data as pdr

# Setting the directory
print(os.getcwd())
new_directory_path = "/Users/hiro/documents/github/Wise Capital" #Change to your directory
os.chdir(new_directory_path)
print(os.getcwd())


OUTPUT_DIR = 'Plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style('whitegrid')

data = pd.read_csv(f'data/Wise_Capital_Data.csv', index_col=0, parse_dates=True)

# =========================================================================================================
# Stationarity Tests
# =========================================================================================================


def test_stationarity(series, name):
    """Run ADF and KPSS tests for stationarity."""
    # ADF test (H0: unit root exists, i.e., non-stationary)
    adf_result = adfuller(series.dropna(), autolag='AIC')
    print(f"\n{name}")
    print(f"  ADF test:")
    print(f"    Statistic: {adf_result[0]:.4f}")
    print(f"    p-value:   {adf_result[1]:.4f}")
    print(f"    Result:    {'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'}")
    
    # KPSS test (H0: series is stationary)
    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    print(f"  KPSS test:")
    print(f"    Statistic: {kpss_result[0]:.4f}")
    print(f"    p-value:   {kpss_result[1]:.4f}")
    print(f"    Result:    {'Stationary' if kpss_result[1] > 0.05 else 'Non-stationary'}")

test_stationarity(data['returns'], "S&P 500 Returns")
test_stationarity(data['sunspot'], "Sunspot Numbers")

# Non-stationary! Let's add some 

##################################################################################################################################


# 1. LINEAR REGRESSION ANALYSIS


y = data["returns"]
X1 = sm.add_constant(data["sunspot"])   # adds intercept

m1 = sm.OLS(y, X1).fit(cov_type="HAC", cov_kwds={"maxlags": 12})  # Newey–West for monthly data
print("Baseline model: S&P 500 returns ~ Sunspots (Newey–West SE)")
print(m1.summary())

beta1 = m1.params["sunspot"] # Beta = 0.0000070
se1 = m1.bse["sunspot"] #sd(Beta) = 0.00005 #Not significant
print(m1.rsquared) # R^2 = 0.0062. Very low 
ci = m1.conf_int().loc["sunspot"]
print("95% CI:", ci.values)


plt.figure(figsize=(9, 5))
plt.scatter(data["sunspot"], y, alpha=0.6, edgecolors="black", linewidths=0.3)
x_grid = np.linspace(data["sunspot"].min(), data["sunspot"].max(), 200)
y_hat = m1.params["const"] + m1.params["sunspot"] * x_grid
plt.plot(x_grid, y_hat, linewidth=2)
plt.axhline(0, linewidth=1, alpha=0.5)
plt.title("Sunspots vs S&P 500 Monthly Log Returns (Baseline)")
plt.xlabel("Sunspot number")
plt.ylabel("S&P 500 monthly log return")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sunspots_vs_returns_baseline.png", dpi=300)
plt.show()

# Result: doesn't seem to make much of a difference. 


# 2. More Controls

controls = [
    "tb3m",
    "yield_spread",
    "credit_spread",
    "inflation_yoy",
    "ip_growth",
    "unemployment",
    "consumer_sent",
    "vix",
]

reg2 = data[["returns", "sunspot"] + controls].dropna()
y2 = reg2["returns"]
X2 = sm.add_constant(reg2[["sunspot"] + controls])

m2 = sm.OLS(y2, X2).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

beta2 = m2.params["sunspot"] #beta = 0.0001
se2 = m2.bse["sunspot"] #sd(beta_2) = 0.00005
ci2 = m2.conf_int().loc["sunspot"].values #[-6.86532707e-06,  2.18383802e-04]


print("\n" + "=" * 80)
print("Controlled regression: returns ~ Sunspots + Controls")
print("=" * 80)
print(f"Beta (sunspot): {beta2:.8f}")
print(f"SE(beta):       {se2:.8f}")
print(f"95% CI:         [{ci2[0]:.8f}, {ci2[1]:.8f}]")
print(f"R²:             {m2.rsquared:.4f}")


# 3. Comparison
print(f"  Baseline:  {beta1:.8f} (SE {se1:.8f})")
print(f"  Controls:  {beta2:.8f} (SE {se2:.8f})")

