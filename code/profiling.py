import matplotlib.pyplot as plt

"""
import sweetviz

plt.rcParams['font.sans-serif'] = ['DFKai-SB']
plt.rcParams['axes.unicode_minus']=False

data = pd.read_csv("../pre_dataset/final.csv")

report = sweetviz.analyze([data, "Data"])
report.show_html("../log/report.html")
"""

import pandas as pd
from pandas_profiling import ProfileReport


df = pd.read_csv("../pre_dataset/final.csv")
profile = ProfileReport(df, minimal=True)
profile.to_file(output_file="output.html")