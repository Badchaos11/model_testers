import utils
from models.FamaFrench5 import FamaFrenchFive
from empyrial import Engine, empyrial
import pandas as pd
import gspread as gd


if __name__ == "__main__":
    """
        Программа предназначена для проведения анализа портфеля, собранного при помощи различных спообов.
        
        Для предсказания доходности портфеля применяется модель Фама-Френча с 5-ю параметрами. Далее производится
    расчёт по Блэку-Литтерману. После всех необходимых расчётов производится предсказание весов
    для каждой компании в портфеле.
        Доступны следующие способы: критерий Келли, Максимальная квадратичная польза, портфель с
    минимальной волатильностью, портфель с максимальным значением рейтинга Шарпа, 
    портфель с минимальной волаильностью при помощи CLA и порфель с максимальным рейтингом Шарпа по CLA.
    
        Праметры класса FamaFrenchFive:
        
        assets - названия компаний, из которых собирается портфель.
        benchmark_ticker - индекс рынка для сравнения доходноси портфеля и всего рынка в целом.
        lookback - количество лет для анализа и предсказания доходности.
        max_size - максимальный процент количества акций одной компании в портфеле.
        min_size - минимальный процент количеств акций одной компании в портфеле.
        test_year - год, для которого производится анализ доходности. Не используеся для расчёта весов, только для 
                    анализа доходности портфеля и индекса. По умолчанию используется 2021-12-31 (2021 год)
        opt_delta - оптимальная дельта, параметр для нахождения весов по методу Max Quad Utility. По умолчанию 1
        risk_free - безрисковая доходность, по умолчанию равна 0.008
        budget - бюджет портфеля, по умолчанию 200000 USD.
        n_iterations - число итераций при нахождении весов по критерию Келли. Слишком большое значение может привести
                       к ошибке и прекращению работы программы, не рекомендуется больше 30, по умолчанию 20.
        plot_res - построение графиков по промежуточным результатам и итогу. По умолчанию False
                       
        В версии 1.0 доступны 2 функции.
        calculate_weights - расчёт весов всеми доступными способами, резултат записывается в файл 
                            portfolio_weight_results.csv
        portfolio_calculate - расчёт доходности портфеля и некоторых параметров для оценки его эффективности. 
                              Для сравнения с портфелем аналогичные расчёты проводятся для индекса.
                              Доступные значения переменной weight_type: 
                              Kelly, Max Quad Util, Max Sharpe, CLA Max Sharpe ,Min Vol ,CLA Min Vol
    """

    assets = ['INTU', 'TXN', 'NTES', 'AEP', 'LRCX', 'PEP', 'AMAT', 'TSLA', 'CTAS', 'TCOM', 'EBAY', 'GOOG', 'CTSH', 'AMGN', 'MTCH', 'CMCSA']
    lkb = 1
    bt = 'QQQ'
    tst = FamaFrenchFive(assets=assets, benchmark_ticker=bt, lookback=lkb,
                         max_size=0.35, min_size=0.0, test_year=2021)
    tst.calculate_weights()
    list_testers = ['Kelly', 'Max Quad Util', 'Max Sharpe', 'CLA Max Sharpe', 'Min Vol']

    # gc = gd.service_account('options-349716-50a9f6e13067.json')
    # worksheet = gc.open('Тесты бэктестинга').worksheet('Бэктест 4.0')
    # n = 156
    #
    # for name in list_testers:
    #     res = tst.portfolio_calculate(name)
    #     try:
    #         cells = utils.insert_colls[res[0]]
    #         print(f"insert {res[1]} into {cells[0]}{n}")
    #         worksheet.update(f"{cells[0]}{n}", res[1])
    #         print(f"insert {res[4]} into {cells[1]}{n}")
    #         worksheet.update(f"{cells[1]}{n}", res[4])
    #         print(f"insert {res[2]} into {cells[2]}{n}")
    #         worksheet.update(f"{cells[2]}{n}", res[2])
    #         print(f"insert {res[3]} into {cells[3]}{n}")
    #         worksheet.update(f"{cells[3]}{n}", res[3])
    #     except TypeError:
    #         print(f'Для данного портфолио не рассчитан вес по {name}')




