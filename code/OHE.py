import pandas as pd

df = pd.read_csv('train.csv')

df_one_hot = pd.get_dummies(df, columns=['年份'])

one_hot_columns = [col for col in df_one_hot if col.startswith('年份_')]
df_one_hot = df_one_hot.reindex

print(df_one_hot)
df_one_hot.to_csv('OCE2.csv')