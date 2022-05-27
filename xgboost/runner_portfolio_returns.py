import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

root_df = pd.read_csv('QQQ.csv').set_index('Unnamed: 0')


def selectFeatures(data_inp, train_labels, numb_of_feat):
    best_features = SelectKBest(score_func=f_regression, k=numb_of_feat)
    fit = best_features.fit(data_inp, train_labels)
    # Select best columns
    cols_KBest_numb = best_features.get_support(indices=True)
    cols_KBest = data_inp.iloc[:, cols_KBest_numb].columns
    return cols_KBest

company_list = []
real_ret = []
pred_ret = []
date_list = []


for company in root_df.index.unique().tolist():
    print('*' * 50)
    print(company)
    for year in range(5):
        # print('*'*50)
        df = root_df.loc[company]
        df = df.replace([np.inf, -np.inf], np.nan).bfill().ffill()

        end_date = str(2020 - year) + '-01-03'  # предсказываем на год вперед
        print(end_date)
        if int(df.Date[0][:4]) + 1 < 2014:

            # print(start_date)

            df_full = df.set_index('Date').copy()
            df = df.set_index('Date').loc[:end_date]
            print(df)

            df_native = df.copy()
            df[['Benchmark_returns_per_year', 'Returns_per_year']] = df[
                ['Benchmark_returns_per_year', 'Returns_per_year']].shift(-252)
            df = df[:-252]  # обрезаем по смещённым данным

            X = df_native[-252:]  # данные фичей которые остались без смещённой доходности

            # Delite columns with only one unique value
            for col in df.columns:
                if len(df[col].unique()) == 1:
                    df.drop(col, inplace=True, axis=1)

            dataset = df.dropna()

            # print(len(dataset))
            # dataset = dataset.drop(['Date'], axis=1)

            #  Replace inf with nan
            dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Extand remaining numbers
            dataset = dataset.bfill(axis=0)
            dataset = dataset.ffill(axis=0)

            # Scale features and select best
            targets_col = ['Returns_per_year']
            data_inp = dataset.drop(targets_col, axis=1)
            data_inp = dataset.drop(['Returns_per_year', 'Benchmark_returns_per_year'], axis=1)
            scaler = StandardScaler()
            scaled_features = StandardScaler().fit_transform(
                dataset.values)  # ТУТ ОШИБКА НУЖНО ПЕРЕДАТЬ С ДРОПНУТЫМИ ПРИЗНАКАМИ
            dataset = pd.DataFrame(scaled_features, index=dataset.index, columns=dataset.columns)
            cols_KBest = selectFeatures(data_inp, dataset[targets_col], numb_of_feat=30)

            # рельная доходность в промежутке времени за который мы пытаемся предсказать доходность, делаем
            # для дальнейшего сравнения
            real_signal = np.where(
                df_full[len(df) + 252:len(df) + 504]['Returns_per_year'] > df_full[len(df) + 252:len(df) + 504][
                    'Benchmark_returns_per_year'].values, 1, 0)

            # accuracy_score(real_signal, pred_signal[0], normalize=True)
            dataset['target'] = np.where(dataset['Returns_per_year'] > dataset['Benchmark_returns_per_year'], 1, 0)
            dataset = dataset.drop(['Returns_per_year', 'Benchmark_returns_per_year'], axis=1)

            col = dataset.columns.tolist()

            # normalize feach per tebele
            scaler_feach = preprocessing.MinMaxScaler(feature_range=(0, 1))
            normal_values = scaler_feach.fit_transform(dataset)
            dataset = pd.DataFrame(normal_values).set_axis(col, axis=1, inplace=False)

            X = X[cols_KBest]

            # нормализация
            col = X.columns.tolist()
            # normalize feach per tebele
            scaler_feach_test = preprocessing.MinMaxScaler(feature_range=(0, 1))
            normal_values_test = scaler_feach_test.fit_transform(X)
            X = pd.DataFrame(normal_values_test).set_axis(col, axis=1, inplace=False)

            X_train, X_test, y_train, y_test = train_test_split(dataset[cols_KBest],
                                                                dataset['target'],
                                                                test_size=0.05,
                                                                shuffle=False)

            # xgboost --------------------------------
            dtrain = xgb.DMatrix(X_train, y_train, silent=True)

            params = {'objective': 'binary:logistic',
                      'max_depth': 7,
                      'eta': 0.8,
                      'verbosity': 0}

            num_rounds = 100

            X_xgb = xgb.DMatrix(X, silent=True)
            xgb_model = xgb.train(params, dtrain, num_boost_round=num_rounds)
            xgb_pred = xgb_model.predict(X_xgb)[-1]
            print(xgb_pred)

            company_list.append(company)
            real_ret.append(real_signal[-1])
            pred_ret.append(xgb_pred)
            date_list.append(int(end_date.split('-')[0]) + 1)

print(pred_ret)
print(real_ret)
output_df = pd.DataFrame({'Company': company_list,
                          'Date': date_list,
                          'Real Return': real_ret,
                          'Predict Return': pred_ret})
print(output_df)
