import pandas as pd
import yfinance as yf
import statsmodels.api as sm

pd.options.mode.chained_assignment = None


def foma_fench_calc(start, assets):
    end = str(start - 1) + '-12-30'
    start = str(start - 1) + '-01-01'  # брать -1 год от даты на начло расчета доходности

    # start = '2018-01-01'
    # end = '2021-12-30'

    # assets = ['AAPL', 'NVDA', 'ADSK', 'AEP', 'MU', 'PEP', 'ISRG', 'FISV']
    benchmark = 'QQQ'
    data_assets = yf.download(assets, start, end)['Close']
    data_bench = yf.download(benchmark, start, end)['Close']

    # доходности
    data = data_assets.pct_change().dropna()
    print(data)
    bench = data_bench.pct_change().dropna()
    print(bench)
    # читаем табицу факторов ежедневных ----------------------------------------------------------------------------

    research_data = pd.read_excel("F-F_Research_Data_5_Factors_2x3_daily.xlsx")
    research_data['Date'] = pd.to_datetime(research_data['Unnamed: 0'], format='%Y%m%d')
    research_data = research_data.drop(columns=['Unnamed: 0'])
    research_data = research_data.set_index('Date')
    research_data = research_data.loc[data.index[0]: data.index[-1]]
    for i in range(len(research_data)):
        research_data['Mkt-RF'][i] = float(research_data['Mkt-RF'][i][:-1])
        research_data['SMB'][i] = float(research_data['SMB'][i][:-1])
        research_data['HML'][i] = float(research_data['HML'][i][:-1])
        research_data['RMW'][i] = float(research_data['RMW'][i][:-1])
        research_data['CMA'][i] = float(research_data['CMA'][i][:-1])
    print(research_data)

    # Получаем таблицу с бетами -----------------------------------------------------------------------------------
    print('Обрабатываю даты')
    bench_df = pd.DataFrame()
    bench_df['Benchmark'] = bench

    tiker_with_factors = research_data.merge(data, how='right', on=['Date']).bfill(axis='rows')
    print('Tick')
    print(tiker_with_factors)

    # Independent variables
    pre_Y = bench_df.merge(tiker_with_factors, how='right', on=['Date']).bfill(axis='rows')
    print(pre_Y)
    # print(Y)
    pre_Y['Mkt-RF'] = (pre_Y['Benchmark'] - pre_Y['RF']) * 100
    print(pre_Y['Mkt-RF'])

    X = pre_Y[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']] / 100  # Mkt-RF =  Benchmark - RF
    print(X)

    # Dependent variable
    Y = data
    print('Треню модель')
    # Create a regression model
    reg = sm.OLS(Y.astype(float), X.astype(float)).fit()

    regression_df = reg.params
    regression_df = regression_df.T
    regression_df['Tickers'] = data.columns.tolist()
    regression_df = regression_df.set_index('Tickers')
    regression_df = regression_df.T
    print('reg df')
    print(regression_df)

    # читаем табицу факторов годовых ----------------------------------------------------------------------------

    research_data_per_year = pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv')
    research_data_per_year = research_data_per_year.rename({'Unnamed: 0': 'Date'}, axis=1)
    research_data_per_year['Date'] = research_data_per_year['Date'].astype('str')
    research_data_per_year = research_data_per_year.set_index('Date') / 100
    print('yd')
    print(research_data_per_year)

    # получаем предикты  --------------------------------------------------------------------------

    low = []
    mid = []
    high = []
    tik = []

    bench_year_return = (data_bench[-1] - data_bench[0]) / data_bench[0]

    for company in regression_df.columns.tolist():
        calk_df = regression_df[company]
        table_data_year = research_data_per_year.loc[end.split('-')[0]]
        try:
            mid_pref = calk_df['Mkt-RF'] * bench_year_return + calk_df['SMB'] * table_data_year['SMB'] + calk_df[
                'HML'] * table_data_year['HML'] + \
                       calk_df['RMW'] * table_data_year['RMW'] + calk_df['CMA'] * table_data_year['CMA']

            mid.append(mid_pref)
            low.append(mid_pref - pre_Y[company].std())
            high.append(mid_pref + pre_Y[company].std())
            tik.append(company)
        except:
            mid.append(0)
            low.append(0)
            high.append(0)
            tik.append(company)

    df = pd.DataFrame()
    df['ticker'] = tik
    df['low_pred_ret'] = low
    df['pred_ret'] = mid
    df['hig_pred_ret'] = high

    out = df.set_index('ticker').T.to_dict('list')

    # обновляем json файл с параметрам, для рассчета весов портфеля  --------------------------------------------------------------------------
    #
    # with open('config.json') as config_file:
    #     data = json.load(config_file)
    #
    # # Заменяем параметры в файле
    # data['views'] = out
    #
    # # Читаем
    # with open("config.json", "w") as outfile:
    #     json.dump(data, outfile)

    print(out)


assets = ['INTU', 'TXN', 'NTES', 'AEP', 'LRCX', 'PEP', 'AMAT', 'TSLA',
          'CTAS', 'TCOM', 'EBAY', 'GOOG', 'CTSH', 'AMGN', 'MTCH', 'CMCSA']
assets = sorted(assets)
print(assets)
year = 2021
foma_fench_calc(year, assets)
