import numpy as np
import pandas as pd

df = pd.read_csv("train.csv")
df_t = pd.DataFrame()

df_t['sum'] = df.loc[df['年份'] == df['年份'],df.columns[4:]].sum(axis=1)

print(df_t)
df_t.to_csv('sum.csv')