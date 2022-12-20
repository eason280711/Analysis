"""
import pandas as pd
import sweetviz as sv

df1 = pd.read_csv("df1.csv")
df2 = pd.read_csv("df2.csv")
df3 = pd.read_csv("df3.csv")
df4 = pd.read_csv("df4.csv")
df5 = pd.read_csv("df5.csv")

# Create a series of boolean values that indicates which dataframe each row belongs to
ls = [True]*5
ls[0] = False
condition_series = pd.Series(ls)

# Concatenate the dataframes and reset the index
df = pd.concat([df1, df2, df3, df4, df5]).reset_index(drop=True)

# Compare the dataframes
report = sv.compare_intra(df, condition_series, names=["df1", "df2", "df3", "df4", "df5"])

# Show the report
report.show_html()
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DFKai-SB']
plt.rcParams['axes.unicode_minus']=False

df = pd.read_csv("../pre_dataset/final.csv")

for i in range(102,110):
    values, counts = np.unique(df.loc[df['年份'] == i,'升學率'].dropna(), return_counts=True)
    plt.bar(values, counts, alpha=0.6, label='{}'.format(i))

plt.xlabel('升學率')
plt.ylabel('數量')
plt.legend(title='年份')

plt.show()