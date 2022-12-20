# 大數據分析

## 概述
高中職學生畢業後的升學、就業、未升學未就業狀況之趨勢分析

## dataset

1. [102~109年 高中職學生升學概況](https://data.gov.tw/dataset/9631)

2. [102~109年 高中職學生就業概況](https://data.gov.tw/dataset/9632)

3. [102~109年 高中職學生未升學、未就業概況](https://data.gov.tw/dataset/9633)

## pre_dataset

- 預處理後的dataset

## log

- step.md

    - 實驗與分析步驟

- output.html

    - pandas_profiling 生成的 全年度ProfileReport

- report.html

    - sweetviz 生成的 全年度ProfileReport

- report_cmp.html

    - pandas_profiling 生成的 102年與103年趨勢比較 ProfileReport

## code

- chart.py

    - 圖表

- check.py

    - 檢查訓練集的列總和是否正常

- model.py

    - 訓練、測試模型
    - 模型效能評估

- OHE.py

    - 對訓練集的輸入做one-hot encoding

- opt.py

    - 自動化調參數

- pre.py

    - 資料集預處理

- progfiling.py

    - EDA與資料分析報告