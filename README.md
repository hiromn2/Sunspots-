# Sunspots

Three main theories link sunspots and economic activity. The first one dates back to Jevons and the Marginalists, saying that those phenomena affect agricultural production, hence, they affect real economic output. The second one is that people coordinate on sunspots or other geomagnetic variables that are irrelevant, but use them as a means to coordinate among different potential equilibria. The final one is a behavioral one. Solar activity may affect human mood, which then could affect aggregate risk preferences in asset prices.

Our objective is to test the third hypothesis using quantitative methods.

# Results

There is **no** evidence for sunspots **to explain** S&P 500 monthly returns in any meaningful way in this sample. Your baseline estimate is essentially zero (β ≈ 0.000007), and is not statistically distinguishable from zero. R² around 0.006 means sunspots account for well under 1% of the variation in monthly returns. Added macro controls as a robustness step, but the results are still the same.

Now, sunspots are observed solar activity and may not reflect how the USA is affected by the sun. Specifically, other climate factors may contribute more strongly to the worker's mood (and hence their financial behavior), which is to say this exercise may not be ideal to determine that no solar activity affects how agents move. Now, it seems reasonable to think that other factors play a bigger role in determining financial returns. Alternatively, we could think about other specifications (a non-linear effect of sunspots into economic activity).

## Project Structure

```text
Sunspots-/
├── README.md
├── requirements.txt
├── Wise_Capital.py
├── Data_Preprocessing.py
│
├── data/
│   ├── SN_m_tot_V2.0.csv
│   ├── sp500.csv
│   └── Wise_Capital_Data.csv   # clean merged dataset
│
└── outputs/
    ├── plots/
    └── results/
