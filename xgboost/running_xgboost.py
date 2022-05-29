import warnings

from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def selectFeatures(data_inp, train_labels, numb_of_feat):
    best_features = SelectKBest(score_func=f_regression, k=numb_of_feat)
    fit = best_features.fit(data_inp, train_labels)
    # Select best columns
    cols_KBest_numb = best_features.get_support(indices=True)
    cols_KBest = data_inp.iloc[:, cols_KBest_numb].columns
    return cols_KBest


def backtester(tickers, year, initial_year):
    s = datetime.strptime(year, '%Y-%m-%d').date()
    s = s + timedelta(days=1)
    start = datetime.strftime(s, '%Y-%m-%d')
    end = f"{initial_year}{start[4:]}"
    print(f"start {start}")
    print(f"end {end}")
    if end[5:] == '02-29':
        print('Fail to load data')
        return None

    df = yf.download(tickers, start, end)['Close'].dropna()
    index_df = yf.download('QQQ', start, end)['Close'].dropna()

    budget = 200000
    b = []
    for i in range(len(tickers)):
        b.append(budget * 1/(len(tickers)))

    cum_benchmark_returns = index_df.pct_change().dropna().cumsum()

    first_sum_portfolio = sum((np.array(b) // np.array(df.iloc[2])) * np.array(df.iloc[2]))
    current_sum_portfolio = sum((np.array(b) // np.array(df.iloc[2])) * np.array(df.iloc[-1]))
    growth_portfolio = (current_sum_portfolio - first_sum_portfolio) / first_sum_portfolio

    portfolio_returns = df.pct_change().dropna()
    portfolio_returns = portfolio_returns.mean(axis=1)
    index_returns = index_df.pct_change().dropna()

    # shp = ep.stats.sharpe_ratio(portfolio_returns, annualization=252)
    # sort = ep.stats.sortino_ratio(portfolio_returns, annualization=252)
    # shp_b = ep.stats.sharpe_ratio(index_returns, annualization=252)
    # sort_b = ep.stats.sortino_ratio(index_returns, annualization=252)
    # inf = ep.stats.excess_sharpe(portfolio_returns,
    #                              index_returns)
    # dd = ep.stats.max_drawdown(portfolio_returns)
    # dd_b = ep.stats.max_drawdown(index_returns)

    print('********************************')
    print(f"Доходность портфеля: {round(growth_portfolio * 100, 2)}%")
    print(f"Доходность портфеля годовая: {round(((growth_portfolio * 100) / len(df)) * 252, 2)}%")
    # print(f"Sharpe ratio: {round(shp, 4)}")
    # print(f"Sortino ratio: {round(sort, 4)}")
    # print(f"Information ratio: {round(inf, 4)}")
    # print(f"Max Drawdown: {round((dd * 100), 2)}%")
    print("*******************************")
    print("****Результат анализа рынка****")
    print(f"Доходность индекса: {round((cum_benchmark_returns[-1] * 100), 2)}%")
    print(f"Доходность индекса годовая: {round(((cum_benchmark_returns[-1] * 100) / len(df)) * 252, 2)}%")
    # print(f"Sharpe ratio for benchmark: {round(shp_b, 4)}")
    # print(f"Sortino ratio for benchmark: {round(sort_b, 4)}")
    # print(f"Max Drawdown benchmark: {round((dd_b * 100), 2)}%")
    print('*******************************')

    # return [round(growth_portfolio * 100, 2), round(shp, 4), round(sort, 4), round((dd * 100), 2)]
    return [end, round(growth_portfolio * 100, 2), round(cum_benchmark_returns[-1] * 100, 2)]


def xg_running(year):
    print(datetime.now())
    root_df = pd.read_csv('QQQ.csv').set_index('Unnamed: 0')
    print(root_df.index.unique().tolist())
    year_date = datetime.strptime(f"{year-1}-01-01", '%Y-%m-%d').date()
    ret_res = []
    date_list = []
    index_ret = []
    companies = []
    a = 0
    for i in range(0, 366):
        date_b_predict = year_date - timedelta(days=1) + timedelta(days=i)
        date_b_predict = datetime.strftime(date_b_predict, '%Y-%m-%d')
        company_list = []
        pred_ret = []
        print(f"Начинаю предсказывать индексы доходности на день после {date_b_predict}. Время: {datetime.now()}")
        for company in root_df.index.unique().tolist():
            df = root_df.loc[company]
            df = df.replace([np.inf, -np.inf], np.nan).bfill().ffill()
            df_full = df.set_index('Date').copy()
            df = df.set_index('Date').loc[:date_b_predict]
            try:
                if df.index[-1] == date_b_predict and len(df) >= 504:
                    df_native = df.copy()
                    df[['Benchmark_returns_per_year', 'Returns_per_year']] = df[
                        ['Benchmark_returns_per_year', 'Returns_per_year']].shift(-252)
                    df = df[:-252]

                    X = df_native[-252:]

                    for col in df.columns:
                        if len(df[col].unique()) == 1:
                            df.drop(col, inplace=True, axis=1)

                    dataset = df.dropna()
                    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
                    dataset = dataset.bfill(axis=0)
                    dataset = dataset.ffill(axis=0)

                    targets_col = ['Returns_per_year']
                    data_inp = dataset.drop(targets_col, axis=1)
                    data_inp = dataset.drop(['Returns_per_year', 'Benchmark_returns_per_year'], axis=1)
                    scaler = StandardScaler()
                    try:
                        scaled_features = StandardScaler().fit_transform(dataset.values)
                    except:
                        break
                    dataset = pd.DataFrame(scaled_features, index=dataset.index, columns=dataset.columns)
                    cols_KBest = selectFeatures(data_inp, dataset[targets_col], numb_of_feat=30)

                    real_signal = np.where(
                        df_full[len(df) + 252:len(df) + 504]['Returns_per_year'] > df_full[len(df) + 252:len(df) + 504][
                            'Benchmark_returns_per_year'].values, 1, 0)
                    dataset['target'] = np.where(dataset['Returns_per_year'] > dataset['Benchmark_returns_per_year'], 1,
                                                 0)
                    dataset = dataset.drop(['Returns_per_year', 'Benchmark_returns_per_year'], axis=1)
                    col = dataset.columns.tolist()
                    scaler_feach = preprocessing.MinMaxScaler(feature_range=(0, 1))
                    normal_values = scaler_feach.fit_transform(dataset)
                    dataset = pd.DataFrame(normal_values).set_axis(col, axis=1, inplace=False)

                    X = X[cols_KBest]

                    col = X.columns.tolist()
                    scaler_feach_test = preprocessing.MinMaxScaler(feature_range=(0, 1))
                    normal_values_test = scaler_feach_test.fit_transform(X)
                    X = pd.DataFrame(normal_values_test).set_axis(col, axis=1, inplace=False)

                    X_train, X_test, y_train, y_test = train_test_split(dataset[cols_KBest],
                                                                        dataset['target'],
                                                                        test_size=0.05,
                                                                        shuffle=False)

                    dtrain = xgb.DMatrix(X_train, y_train, silent=True)

                    params = {'objective': 'binary:logistic',
                              'max_depth': 7,
                              'eta': 0.8,
                              'verbosity': 0}

                    num_rounds = 100

                    X_xgb = xgb.DMatrix(X, silent=True)
                    xgb_model = xgb.train(params, dtrain, num_boost_round=num_rounds)
                    xgb_pred = xgb_model.predict(X_xgb)[-1]
                    company_list.append(company)
                    pred_ret.append(xgb_pred)
                    print('.', end='', flush=True)

                else:
                    if df.index[-1] != date_b_predict:
                        print(f'{date_b_predict} торговли не было, переходим к следующей дате')
                        break
            except IndexError:
                pass
        print('.')
        print(50 * '*')
        tdf = pd.DataFrame()
        tdf['Companies'] = company_list
        tdf['Returns Pred'] = pred_ret
        tdf = tdf.set_index('Companies')
        tdf = tdf.sort_values(by=['Returns Pred'])
        if len(tdf) == 0:
            continue
        print('Топ 16 предсказанных компаний по доходности')
        x = tdf.iloc[-16:].index.tolist()
        for j in range(len(x)):
            x[j] = x[j][4:]
        print(x)
        print(50 * '*')
        resu = backtester(x, date_b_predict, year)
        if resu is None:
            continue
        companies.append(x)
        date_list.append(resu[0])
        ret_res.append(resu[1])
        index_ret.append(resu[2])

    res_df = pd.DataFrame()
    res_df['Portfolio'] = companies
    res_df['Date'] = date_list
    res_df['Portfolio Returns'] = ret_res
    res_df['Index Returns'] = index_ret
    res_df = res_df.set_index('Date')
    res_df.to_csv('Test_XGB_Run.csv')
    print(res_df)
    print(datetime.now())


xg_running(2021)
