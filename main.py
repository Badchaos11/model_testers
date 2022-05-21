from models.FamaFrench5 import FamaFrenchFive


if __name__ == "__main__":
    assets = ['INTU', 'TXN', 'NTES', 'AEP', 'LRCX', 'PEP', 'AMAT', 'TSLA', 'CTAS', 'TCOM', 'EBAY', 'GOOG']
    lkb = 2
    bt = 'QQQ'
    tst = FamaFrenchFive(assets=assets, benchmark_ticker=bt, lookback=lkb,
                         max_size=0.35, min_size=0)
    tst.calculate_weights()
