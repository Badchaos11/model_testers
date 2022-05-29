import pandas as pd
import numpy as np

df = pd.read_csv('Test_XGB_Run.csv')
print(df)
df = df.set_index('Date')

df['Over Index'] = np.where(df['Portfolio Returns'] > df['Index Returns'], 1, 0)

df.to_csv('Test_XGB_Run_With_Overincome.csv')
