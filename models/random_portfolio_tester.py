import random

import empyrical as ep
import numpy as np
import yfinance as yf
import gspread as gd
from utils import nasdaq_list, ew


def backtester(tickers, year):
    start = f"{year}-01-01"
    end = f"{year}-12-30"

    df = yf.download(tickers, start, end)['Close'].dropna()
    index_df = yf.download('QQQ', start, end)['Close'].dropna()

    budget = 200000
    b = []
    for i in range(len(tickers)):
        b.append(budget * 1/(len(tickers)))
    print(b)

    cum_benchmark_returns = index_df.pct_change().dropna().cumsum()

    first_sum_portfolio = sum((np.array(b) // np.array(df.iloc[2])) * np.array(df.iloc[2]))
    current_sum_portfolio = sum((np.array(b) // np.array(df.iloc[2])) * np.array(df.iloc[-1]))
    growth_portfolio = (current_sum_portfolio - first_sum_portfolio) / first_sum_portfolio

    portfolio_returns = df.pct_change().dropna()
    portfolio_returns = portfolio_returns.mean(axis=1)
    index_returns = index_df.pct_change().dropna()

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

    return [round(growth_portfolio * 100, 2), round(shp, 4), round(sort, 4), round((dd * 100), 2)]


# nasdaq_list = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'GOOG', 'FB', 'ADBE', 'NFLX', 'CMCSA', 'CSCO', 'COST',
#                'AVGO', 'PEP', 'PYPL', 'INTC', 'QCOM', 'TXN', 'INTU', 'AMD', 'TMUS', 'HON', 'AMAT', 'SBUX', 'CHTR',
#                'MRNA', 'AMGN', 'ISRG', 'ADP', 'ADI', 'LRCX', 'MU', 'GILD', 'BKNG', 'MDLZ', 'CSX', 'MRVL', 'REGN',
#                'FISV', 'ASML', 'JD', 'KLAC', 'NXPI', 'ADSK', 'LULU', 'ILMN', 'XLNX', 'VRTX', 'SNPS', 'MELI', 'EXC',
#                'WDAY', 'DXCM', 'IDXX', 'CDNS', 'ALGN', 'KDP', 'MAR', 'TEAM', 'MCHP', 'ORLY', 'ATVI', 'ZM', 'MNST',
#                'CTAS', 'EBAY', 'PAYX', 'CTSH', 'AEP', 'KHC', 'WBA', 'ROST', 'CRWD', 'VRSK', 'EA', 'XEL', 'MTCH',
#                'BIDU', 'FAST', 'CPRT', 'ANSS', 'BIIB', 'DLTR', 'OKTA', 'NTES', 'PCAR', 'SGEN', 'VRSN', 'CDW', 'DOCU',
#                'SIRI', 'SWKS', 'CERN', 'PDD', 'SPLK', 'INCY', 'CHKP', 'TCOM', 'PTON', 'FOXA', 'FOX', 'AZN']

# assets = sorted(random.sample(nasdaq_list, 10))
# year = 2021
# backtester(assets, year)
gc = gd.service_account('../options-349716-50a9f6e13067.json')
worksheet = gc.open('Тесты бэктестинга').worksheet('Бэктест 4.0')

asts = ['QCOM', 'AZN', 'MNST', 'ROST', 'CERN', 'ORLY', 'MELI', 'CHTR', 'LULU', 'GOOG', 'VRSN', 'VRSK', 'VRTX', 'ATVI', 'AVGO', 'SBUX']
m = 126
res = backtester(asts, year=2018)
print(f"insert {res[0]} into {ew[0]}{m}")
worksheet.update(f"{ew[0]}{m}", res[0])
print(f"insert {res[3]} into {ew[1]}{m}")
worksheet.update(f"{ew[1]}{m}", res[3])
print(f"insert {res[1]} into {ew[2]}{m}")
worksheet.update(f"{ew[2]}{m}", res[1])
print(f"insert {res[2]} into {ew[3]}{m}")
worksheet.update(f"{ew[3]}{m}", res[2])
