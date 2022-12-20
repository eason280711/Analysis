import matplotlib.pyplot as plt

"""
import sweetviz

plt.rcParams['font.sans-serif'] = ['DFKai-SB']
plt.rcParams['axes.unicode_minus']=False

data = pd.read_csv("../pre_dataset/final.csv")

report = sweetviz.analyze([data, "Data"])
report.show_html("../log/report.html")
"""

"""
import pandas as pd
from pandas_profiling import ProfileReport


df = pd.read_csv("../pre_dataset/final.csv")
profile = ProfileReport(df, minimal=True)
profile.to_file(output_file="output.html")
"""
import pandas as pd
import sweetviz

plt.rcParams['font.sans-serif'] = ['DFKai-SB']
plt.rcParams['axes.unicode_minus']=False

df1 = pd.read_csv("109.csv")
df2 = pd.read_csv("110.csv")

report = sweetviz.compare([df1, "109"], [df2, "110"])

report.show_html("../log/102-110cmp.html")