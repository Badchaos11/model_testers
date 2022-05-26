import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import riskfolio as rp
import matplotlib.pyplot as plt
import quantstats as qs
import empyrical as ep


def hierarchial_risk_parity_calc_returns(assets, year, rms, lookback=3, bench='QQQ'):

    data = yf.download(assets, start=f"{year-lookback}-01-01", end=f"{year-1}-12-30", progress=False)['Close']
    Y = data.pct_change().dropna()
    port = rp.HCPortfolio(returns=Y)
    w_s = pd.DataFrame([])

    model = 'HERC'
    codependence = 'pearson'
    rf = 0
    linkage = 'single'
    max_k = 10
    leaf_order = True

    for m in rms:
        w = port.optimization(model=model,
                              codependence=codependence,
                              rm=m,
                              rf=rf,
                              linkage=linkage,
                              max_k=max_k,
                              leaf_order=leaf_order)

        w_s = pd.concat([w_s, w], axis=1)

    w_s.columns = rms
    for r in rms:
        print(f"for {r}")
        print(w_s[r])
        print('\n')




assets = ['INTU', 'TXN', 'NTES', 'AEP', 'LRCX', 'PEP', 'AMAT', 'TSLA', 'CTAS', 'TCOM',
          'EBAY', 'GOOG', 'CTSH', 'AMGN', 'MTCH', 'CMCSA']
assets = sorted(assets)
year = 2021
rms = ['vol', 'MV', 'MAD', 'MSV', 'FLPM', 'SLPM',
       'VaR', 'CVaR', 'WR', 'MDD']

hierarchial_risk_parity_calc_returns(assets, year, rms)
