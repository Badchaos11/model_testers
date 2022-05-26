import random
from utils import nasdaq_list
import pandas as pd
import empyrical as ep
import numpy as np
import yfinance as yf


def backtester(tickers, year):
    start = f"{year}-01-01"
    end = f"{year}-12-30"

    df = yf.download(tickers, start, end)['Close'].dropna()
    index_df = yf.download('QQQ', start, end)['Close'].dropna()

    prices = df.iloc[0].reindex(tickers)
    print(prices)
    print(prices[0])
    weights = []
    for i in range(len(prices)):
        a = round(prices.max() / prices[i]) * 1
        weights.append(a)

    quantity = weights

    cum_benchmark_returns = index_df.pct_change().dropna().cumsum()

    first_sum_portfolio = sum(df.iloc[2] * quantity)
    current_sum_portfolio = sum(df.iloc[-1] * quantity)
    growth_portfolio = (current_sum_portfolio - first_sum_portfolio) / first_sum_portfolio

    portfolio_returns = (df * quantity).sum(axis=1).pct_change().dropna()
    index_returns = index_df.pct_change().dropna()

    risk_free_rate = 0
    shp = ep.stats.sharpe_ratio(portfolio_returns, annualization=252)
    sort = ep.stats.sortino_ratio(portfolio_returns, annualization=252)
    shp_b = ep.stats.sharpe_ratio(index_returns, annualization=252)
    sort_b = ep.stats.sortino_ratio(index_returns, annualization=252)
    inf = ep.stats.excess_sharpe(portfolio_returns,
                                 index_returns)
    dd = ep.stats.max_drawdown(portfolio_returns)
    dd_b = ep.stats.max_drawdown(index_returns)

    print(f"Доходность портфееля {tickers} при равных весах")
    print('********************************')
    print(f"Доходность портфеля: {round(growth_portfolio * 100, 2)}%")
    print(f"Доходность портфеля годовая: {round(((growth_portfolio * 100) / len(df)) * 252, 2)}%")
    print(f"Sharpe ratio: {round(shp, 4)}")
    print(f"Sortino ratio: {round(sort, 4)}")
    print(f"Information ratio: {round(inf, 4)}")
    print(f"Max Drawdown: {round((dd * 100), 2)}%")
    print("*******************************")
    print("****Результат анализа рынка****")
    print(f"Доходность индекса: {round((cum_benchmark_returns[-1] * 100), 2)}%")
    print(f"Доходность индекса годовая: {round(((cum_benchmark_returns[-1] * 100) / len(df)) * 252, 2)}%")
    print(f"Sharpe ratio for benchmark: {round(shp_b, 4)}")
    print(f"Sortino ratio for benchmark: {round(sort_b, 4)}")
    print(f"Max Drawdown benchmark: {round((dd_b * 100), 2)}%")
    print('*******************************')


assets = sorted(random.sample(nasdaq_list, 10))
year = 2021
backtester(assets, year)
