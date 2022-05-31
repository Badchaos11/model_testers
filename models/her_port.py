import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import riskfolio as rp
import matplotlib.pyplot as plt
import quantstats as qs
import empyrical as ep


warnings.filterwarnings("ignore")

start = '2013'
end = '2018'

# Tickers of assets
assets = ['MAR', 'ASML', 'DFS', 'WU', 'KMX', 'PFG', 'LEG', 'LULU', 'AXP', 'CTRA', 'CSX', 'AVGO', 'PPG', 'CF', 'COF', 'FCX', 'ETN', 'SIVB', 'ZBRA', 'WDC', 'PKG', 'SYK', 'AMP', 'NSC', 'WMB', 'NEM', 'MSCI', 'EMN', 'CERN', 'LYV', 'RMD', 'CMG', 'RHI', 'GS', 'FLT', 'PH']
assets = sorted(assets)

# Downloading data
data = yf.download(assets)
data = data.loc[:, ('Adj Close', slice(None))]
data.columns = assets
# Calculating returns
Y = data.pct_change().dropna()

test_df = pd.DataFrame()


def backtest(stock, weights, bench='QQQ', start_date='2017-01-01', end_date='2017-12-30'):

    if len(stock) > 1:
        assets_prices = yf.download(stock, start_date, end_date, progress=False)['Close']
        assets_prices = assets_prices.filter(stock)
        returns = assets_prices.pct_change()[1:]
        returns_weighted = (returns * weights).sum(axis=1)

    cag = ep.stats.cagr(returns_weighted, period='daily', annualization=None)
    cag = str(round(cag * 100, 2))

    sp = qs.stats.sharpe(returns_weighted, rf=0)
    sp = str(np.round(sp, decimals=4))

    sor = ep.stats.sortino_ratio(returns_weighted, required_return=0, period='daily')
    sor = str(round(sor, 2))

    mdd = ep.stats.max_drawdown(returns_weighted, out=None)
    mdd = str(round(mdd * 100, 2))

    print(f"Strategy results for MAD weights:\n"
          f"Annual returns: {cag}%;\n"
          f"Sharpe ratio: {sp};\n"
          f"Sortino ratio: {sor};\n"
          f"Max drawdown: {mdd}%;\n")


for i in range(0, (int(end) - int(start))):
    data2 = data[str(int(start) + i):str(int(start) + 3 + i)]
    Y2 = Y[str(int(start) + i):str(int(start) + 3 + i)]
    port = rp.HCPortfolio(returns=Y2)

    # Estimate optimal portfolio:
    model = 'HERC'  # Could be HRP or HERC
    codependence = 'pearson'  # Correlation matrix used to group assets in clusters
    rm = 'MV'  # Risk measure used, this time will be variance
    rf = 0  # Risk free rate
    linkage = 'single'  # Linkage method used to build clusters
    max_k = 10  # Max number of clusters used in two difference gap statistic, only for HERC model
    leaf_order = True  # Consider optimal order of leafs in dendrogram

    rms = ['vol', 'MV', 'MAD', 'MSV', 'FLPM', 'SLPM',
           'VaR', 'CVaR', 'WR', 'MDD']

    w_s = pd.DataFrame([])

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

    budget = 1000000
    ret = []

    for n in range(len(w_s.T)):
        ret.append((((w_s.T.iloc[n] * budget // data[str(int(start) + 4 + i)].iloc[0]) *
                     data[str(int(start) + 4 + i)].iloc[-1]).sum()
                    - ((w_s.T.iloc[n] * budget // data[str(int(start) + 4 + i)].iloc[0]) *
                       data[str(int(start) + 4 + i)].iloc[0]).sum()) /
                   ((w_s.T.iloc[n] * budget // data[str(int(start) + 4 + i)].iloc[0]) *
                    data[str(int(start) + 4 + i)].iloc[0]).sum())
    print(f"turn {i}")
    print(str(int(start) + 4 + i))
    backtest(assets, w_s['MAD'].tolist())
    test_df[str(int(start) + 4 + i)] = ret


test_df['opt'] = w_s.T.index
print(test_df)

lis = test_df['opt'].tolist()
