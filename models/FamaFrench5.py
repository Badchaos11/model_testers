import cvxopt.coneprog
import pandas as pd
import yfinance as yf
import numpy as np
import empyrical as ep
import statsmodels.api as sm
import json
import argparse
import datetime
import matplotlib.pyplot as plt
import statsmodels.stats.moment_helpers as mh
import sys
import json
from cvxopt.solvers import qp
from cvxopt import matrix
from joblib import Memory
from pandas_datareader import data as pd_data
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import EfficientFrontier, objective_functions, CLA

pd.options.mode.chained_assignment = None


class FamaFrenchFive:

    def __init__(self, assets: list, benchmark_ticker: str, lookback: int, max_size: float, min_size: float,
                 test_year="2021-12-31", risk_free=0.008, budget=2e5, n_iterations=20):
        self.__tickers = sorted(assets)
        self.__benchmark_ticker = benchmark_ticker
        self.__lookback = lookback
        self.__max_size = max_size
        self.__min_size = min_size
        self.__risk_free = risk_free
        self.__test_year = test_year
        self.__budget = budget
        self.__currencies = ['CADUSD=X', 'AUDUSD=X', 'TRYUSD=X', 'CHFUSD=X', 'MYRUSD=X', 'MXNUSD=X',
                             'GBPUSD=X', 'SGDUSD=X', 'HKDUSD=X', 'ZARUSD=X', 'ILSUSD=X', 'JPYUSD=X',
                             'RUBUSD=X', 'EURUSD=X']
        cvxopt.coneprog.options = {'maxiters': n_iterations}

    def __load_prices(self):
        ty = datetime.datetime.strptime(self.__test_year, "%Y-%m-%d").date()
        ed = ty - datetime.timedelta(days=365)
        st = ed - datetime.timedelta(days=356 * self.__lookback)
        data = yf.download(self.__tickers, st, ed)['Close'].dropna()

        self.__data = data

    def __load_market_price(self):
        market_prices = yf.download("SPY", period="max")["Adj Close"]
        self.__market_prices = market_prices

    def __load_mkt_caps(self):
        mcaps = pd_data.get_quote_yahoo(self.__tickers)['marketCap']
        missing_mcap_symbols = mcaps[mcaps.isnull()].index
        for symbol in missing_mcap_symbols:
            print('attempting to find market cap info for', symbol)
            data = yf.Ticker(symbol)
            if data.info['quoteType'] == 'ETF' or data.info['quoteType'] == 'MUTUALFUND':
                mcap = data.info['totalAssets']
                print('adding market cap info for', symbol)
                mcaps.loc[symbol] = mcap
            else:
                print('Failed to find market cap for', symbol)
                sys.exit(-1)
        self.__mkt_caps = mcaps

    def __load_mean_views(self):
        mu = {}
        for symbol in sorted(self.__tickers):
            mu[symbol] = self.__views[symbol][1]
        self.__mu = mu

    def __load_full_data(self):
        self.__load_prices()
        self.__load_market_price()
        self.__load_mkt_caps()

    def __calc_omega(self):
        variances = []
        for symbol in sorted(self.__tickers):
            view = self.__views[symbol]
            lb, ub = view[0], view[2]
            std_dev = (ub - lb) / 2
            variances.append(std_dev ** 2)
        omega = np.diag(variances)
        self.__omega = omega

    def __calc_quantity(self, weight_type):
        wgts = pd.read_csv('models/portfolio_weight_results.csv')
        wgts = wgts[weight_type]
        weighted_budget = [self.__budget * wgts[i] for i in range(len(wgts))]
        return weighted_budget

    def __count_fama(self):

        df = self.__data.pct_change().dropna()

        ff_ratios = pd.read_excel("models/F-F_Research_Data_5_Factors_2x3_daily.xlsx")
        ff_ratios['Date'] = pd.to_datetime(ff_ratios['Unnamed: 0'], format='%Y%m%d')
        ff_ratios = ff_ratios.set_index('Date')
        ff_ratios = ff_ratios.drop(columns=['Unnamed: 0'])
        ff_ratios = ff_ratios.loc[df.index[0]: df.index[-1]]
        for i in range(len(ff_ratios)):
            ff_ratios['Mkt-RF'][i] = float(ff_ratios['Mkt-RF'][i][:-1])
            ff_ratios['SMB'][i] = float(ff_ratios['SMB'][i][:-1])
            ff_ratios['HML'][i] = float(ff_ratios['HML'][i][:-1])
            ff_ratios['RMW'][i] = float(ff_ratios['RMW'][i][:-1])
            ff_ratios['CMA'][i] = float(ff_ratios['CMA'][i][:-1])

        dd = pd.read_csv("models/F-F_Research_Data_5_Factors_2x3.csv")
        ed = -1 * self.__lookback - 1
        dd = dd.iloc[ed:-1]
        dd = dd.drop("Unnamed: 0", axis=1)

        low = []
        mid = []
        high = []
        tik = []
        X = ff_ratios.drop('RF', axis=1)

        for i, ticker in enumerate(self.__tickers):
            try:
                y = df[ticker]
                reg = sm.OLS(y, X.astype(float)).fit()

                mid_pref = dd['RF'].mean() / 100 + reg.params[0] * dd["Mkt-RF"].mean() / 100 \
                           + reg.params[1] * dd['SMB'].mean() / 100 \
                           + reg.params[2] * dd['HML'].mean() / 100 \
                           + reg.params[3] * dd['RMW'].mean() / 100 \
                           + reg.params[4] * dd['CMA'].mean() / 100
                mid.append(mid_pref)
                low.append(mid_pref - df[self.__tickers[i]].std())
                high.append(mid_pref + df[self.__tickers[i]].std())
                tik.append(self.__tickers[i])
            except:
                mid.append(0)
                low.append(0)
                high.append(0)
                tik.append(self.__tickers[i])

        df = pd.DataFrame()
        df['ticker'] = tik
        df['low_pred_ret'] = low
        df['pred_ret'] = mid
        df['hig_pred_ret'] = high

        out = df.set_index('ticker').T.to_dict('list')
        self.__views = out

    def __calculate_black_litterman(self):
        delta = black_litterman.market_implied_risk_aversion(self.__market_prices)
        covar = risk_models.risk_matrix(self.__data, method='oracle_approximating')
        market_prior = black_litterman.market_implied_prior_returns(self.__mkt_caps, risk_aversion=delta,
                                                                    cov_matrix=covar)
        self.__calc_omega()
        bl = BlackLittermanModel(covar, pi="market", market_caps=self.__mkt_caps, risk_aversion=delta,
                                 absolute_views=self.__mu, omega=self.__omega)
        rets_bl = bl.bl_returns()
        covar_bl = bl.bl_cov()
        self.__rets_bl = rets_bl
        self.__covar_bl = covar_bl

    def __kelly_optimise(self):
        M = self.__rets_bl.to_numpy()
        C = self.__covar_bl.to_numpy()

        n = M.shape[0]
        A = matrix(1.0, (1, n))
        b = matrix(1.0)
        G = matrix(0.0, (n, n))
        G[::n + 1] = -1.0
        h = matrix(0.0, (n, 1))

        try:
            max_pos_size = self.__min_size
        except KeyError:
            max_pos_size = None
        try:
            min_pos_size = self.__max_size
        except KeyError:
            min_pos_size = None
        if min_pos_size is not None:
            h = matrix(min_pos_size, (n, 1))

        if max_pos_size is not None:
            h_max = matrix(max_pos_size, (n, 1))
            G_max = matrix(0.0, (n, n))
            G_max[::n + 1] = 1.0
            G = matrix(np.vstack((G, G_max)))
            h = matrix(np.vstack((h, h_max)))

        S = matrix((1.0 / ((1 + self.__risk_free) ** 2)) * C)
        q = matrix((1.0 / (1 + self.__risk_free)) * (M - self.__risk_free))
        sol = qp(S, -q, G, h, A, b)
        kelly = np.array([sol['x'][i] for i in range(n)])
        kelly = pd.DataFrame(kelly, index=self.__covar_bl.columns, columns=['Weights'])
        kelly = kelly.round(3)
        kelly.columns = ['Kelly']
        return kelly

    def __max_quad_utility_weights(self):
        print('Begin max quadratic utility optimization')
        returns, sigmas, weights, deltas = [], [], [], []
        for delta in np.arange(1, 10, 1):
            ef = EfficientFrontier(self.__rets_bl, self.__covar_bl,
                                   weight_bounds=(self.__min_size, self.__max_size))
            ef.max_quadratic_utility(delta)
            ret, sigma, __ = ef.portfolio_performance()
            weights_vec = ef.clean_weights()
            returns.append(ret)
            sigmas.append(sigma)
            deltas.append(delta)
            weights.append(weights_vec)
        '''
        fig, ax = plt.subplots()
        ax.plot(sigmas, returns)
        for i, delta in enumerate(deltas):
            ax.annotate(str(delta), (sigmas[i], returns[i]))
        plt.xlabel('Volatility (%) ')
        plt.ylabel('Returns (%)')
        plt.title('Efficient Frontier for Max Quadratic Utility Optimization')
        plt.show()
        '''
        opt_delta = float(input('Enter the desired point on the efficient frontier: '))
        ef = EfficientFrontier(self.__rets_bl, self.__covar_bl,
                               weight_bounds=(self.__min_size, self.__max_size))
        ef.max_quadratic_utility(opt_delta)
        opt_weights = ef.clean_weights()
        opt_weights = pd.DataFrame.from_dict(opt_weights, orient='index')
        opt_weights.columns = ['Max Quad Util']
        return opt_weights, ef

    def __min_volatility_weights(self):
        ef = EfficientFrontier(self.__rets_bl, self.__covar_bl,
                               weight_bounds=(self.__min_size, self.__max_size))
        ef.min_volatility()
        weights = ef.clean_weights()
        weights = pd.DataFrame.from_dict(weights, orient='index')
        weights.columns = ['Min Vol']
        return weights, ef

    def __max_sharpe_weights(self):
        ef = EfficientFrontier(self.__rets_bl, self.__covar_bl,
                               weight_bounds=(self.__min_size, self.__max_size))
        ef.max_sharpe()
        weights = ef.clean_weights()
        weights = pd.DataFrame.from_dict(weights, orient='index')
        weights.columns = ['Max Sharpe']
        return weights, ef

    def __cla_max_sharpe_weights(self):
        cla = CLA(self.__rets_bl, self.__covar_bl,
                  weight_bounds=(self.__min_size, self.__max_size))
        cla.max_sharpe()
        weights = cla.clean_weights()
        weights = pd.DataFrame.from_dict(weights, orient='index')
        weights.columns = ['CLA Max Sharpe']
        return weights, cla

    def __cla_min_vol_weights(self):
        cla = CLA(self.__rets_bl, self.__covar_bl,
                  weight_bounds=(self.__min_size, self.__max_size))
        cla.min_volatility()
        weights = cla.clean_weights()
        weights = pd.DataFrame.from_dict(weights, orient='index')
        weights.columns = ['CLA Min Vol']
        return weights, cla

    def calculate_weights(self):
        print('Начинаю расчитывать веса для портфеля')
        print('Загружаю данные')
        self.__load_full_data()
        print('Рассчитываю модель Фама-Френча с 5 параметрами')
        self.__count_fama()
        self.__load_mean_views()
        print('Рассчитыаю модель Блэка-Литтермана')
        self.__calculate_black_litterman()
        print('Расчет по критерию Kelly')
        kelly_w = self.__kelly_optimise()
        print('Расчет по Max Quad Utility')
        max_quad_util_w, max_quad_util_ef = self.__max_quad_utility_weights()
        print('Расчёт для минимальной волатильности')
        min_vol_w, min_vol_ef = self.__min_volatility_weights()
        print('Расчёт для максимальногео рейтинга Шарпа')
        max_sharpe_w, max_sharpe_ef = self.__max_sharpe_weights()
        print('Расчет для максимального рейтинга Шарпа по CLA')
        cla_max_sharpe_w, cla_max_sharpe_cla = self.__cla_max_sharpe_weights()
        print('Расчёт для минимальной волатильности по CLA')
        cla_min_vol_w, cla_min_vol_cla = self.__cla_min_vol_weights()
        print('Заполнение весов')
        weights_df = pd.merge(kelly_w, max_quad_util_w, left_index=True, right_index=True)
        weights_df = pd.merge(weights_df, max_sharpe_w, left_index=True, right_index=True)
        weights_df = pd.merge(weights_df, cla_max_sharpe_w, left_index=True, right_index=True)
        weights_df = pd.merge(weights_df, min_vol_w, left_index=True, right_index=True)
        weights_df = pd.merge(weights_df, cla_min_vol_w, left_index=True, right_index=True)
        weights_df.to_csv('models/portfolio_weight_results.csv')

        self.__weights = weights_df

    def portfolio_calculate(self, weights_type: str):
        ed = datetime.datetime.strptime(self.__test_year, "%Y-%m-%d")
        st = ed - datetime.timedelta(days=365)
        df = yf.download(self.__tickers, st, ed, progress=False)['Close'].fillna(0)
        index_df = yf.download(self.__benchmark_ticker, st, ed, progress=False)['Close'].fillna(0)
        try:
            b = self.__calc_quantity(weights_type)
            cum_benchmark_returns = index_df.pct_change().dropna().cumsum()
            portfolio_returns = df.pct_change().dropna()
            portfolio_returns = portfolio_returns.mean(axis=1)
            index_returns = index_df.pct_change().dropna()

            first_sum_portfolio = sum((np.array(b) // np.array(df.iloc[2])) * np.array(df.iloc[2]))
            current_sum_portfolio = sum((np.array(b) // np.array(df.iloc[2])) * np.array(df.iloc[-1]))
            growth_portfolio = (current_sum_portfolio - first_sum_portfolio) / first_sum_portfolio

            shp = ep.stats.sharpe_ratio(portfolio_returns, annualization=252)
            sort = ep.stats.sortino_ratio(portfolio_returns, annualization=252)
            shp_b = ep.stats.sharpe_ratio(index_returns, annualization=252)
            sort_b = ep.stats.sortino_ratio(index_returns, annualization=252)
            inf = ep.stats.excess_sharpe(portfolio_returns,
                                         index_returns)
            dd = ep.stats.max_drawdown(portfolio_returns)
            dd_b = ep.stats.max_drawdown(index_returns)

            """
            cumprod_ret = ((df * quantity).sum(axis=1).pct_change().dropna() + 1).cumprod() * 100
            cumprod_market_ret = (index_df.pct_change().dropna() + 1).cumprod() * 100
            cumprod_ret.index = pd.to_datetime(cumprod_ret.index)
            trough_index = (np.maximum.accumulate(cumprod_ret) - cumprod_ret).idxmax()
            peak_index = cumprod_ret.loc[:trough_index].idxmax()
            maximum_drawdown = 100 * (cumprod_ret[trough_index] - cumprod_ret[peak_index]) / cumprod_ret[peak_index]
            """
            print(f"Результат анализа портфеля по {weights_type}")
            print('********************************')
            print('Количество акций в портфеле:')
            for i in range(len(self.__tickers)):
                x = int(b[i] // df[self.__tickers[i]][2])
                print(f"Акций компании {self.__tickers[i]} в портфеле: {x}")
            print('********************************')
            print('Первоначальная стоимость портфеля', round(first_sum_portfolio))
            print('Текущая стоимость портфеля', round(current_sum_portfolio))
            print('***Результат анализа портфеля***')
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
        except:
            print('Такого типа весов несуществует, попробуйте один из этих: \n'
                  'Kelly, Max Quad Util, Max Sharpe, CLA Max Sharpe ,Min Vol ,CLA Min Vol')
