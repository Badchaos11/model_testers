import warnings

from datetime import datetime, timedelta
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


def xg_running(year):
    print(datetime.now())
    root_df = pd.read_csv('QQQ.csv').set_index('Unnamed: 0')
    print(root_df.index.unique().tolist())
    print(len(root_df.index.unique().tolist()))
    result_df = pd.DataFrame()
    year_date = datetime.strptime(f"{year}-01-01", '%Y-%m-%d').date()
    a = 0
    for i in range(0, 366):
        date_b_predict = year_date - timedelta(days=1) + timedelta(days=i)
        date_b_predict = datetime.strftime(date_b_predict, '%Y-%m-%d')
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
                    # print(f"Предсказание для {company} на следующий день после "
                    #       f"{date_b_predict}: {xgb_pred}")

                else:
                    if df.index[-1] != date_b_predict:
                        print(f'{date_b_predict} торговли не было, переходим к следующей дате')
                        # print(f"Не буду предсказывать для {company}, последняя дата в данных {df.index[-1]},"
                        #       f" а дата в цикле {date_b_predict}")
                        a -= 1
                        break
                    # elif len(df) < 504:
                    #     print(f"Нет достаточно данных компании {company} на {date_b_predict} для предсказания,"
                    #           f"требуется минимум 500, а есть {len(df)}")
            except IndexError:
                pass
                # print(f'Нет данных для компании компании {company}')
        a += 1
    print(a)
    print(datetime.now())


xg_running(2020)
