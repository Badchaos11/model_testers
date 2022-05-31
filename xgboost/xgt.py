import pandas as pd
import numpy as np

df = pd.read_csv('_XGB_Run_Daily_2020.csv')
print(df)
df = df.set_index('Date')

df['Over Index'] = np.where(df['Portfolio Returns'] > df['Index Returns'], 1, 0)

df.to_csv('XGB_Run_Daily_2020_Overrun.csv')
