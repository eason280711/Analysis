import pandas as pd
import numpy as np
df = pd.read_csv('OCE.csv')

X = df.iloc[:, 39:]
y = df.iloc[:, :39]

X = X.to_numpy()
y = y.to_numpy()

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import TheilSenRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

# base_estimator = RandomForestRegressor(
#     n_estimators=420,
#     max_depth=20
# )

# base_estimator = MLPRegressor(
#     hidden_layer_sizes=500,  # 隱藏層大小
#     activation="relu",  # 激活函數
#     solver="adam",  # 優化器
#     max_iter=500,  # 最大迭代次數
#     random_state=0,
#     alpha=1
# )

#base_estimator = BayesianRidge()

#base_estimator = TheilSenRegressor()

# base_estimator = LinearRegression(fit_intercept=False)

# DecisionTreeRegressor()
#

#base_estimator = GradientBoostingRegressor(n_estimators=198, learning_rate=0.01126,
#                                max_depth=22, random_state=42,alpha=0.0024477)
base_estimator = GradientBoostingRegressor(n_estimators=10, learning_rate=0.01126,
                                max_depth=3, random_state=42,alpha=0.0024477)

base_estimator = MultiOutputRegressor(base_estimator)

model = base_estimator

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

kfold = KFold(n_splits=2)
for train_index, test_index in kfold.split(X):
    # 取出訓練集和驗證集
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 對訓練集進行訓練
    model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")

print(model.score(X,y,sample_weight=None))

from sklearn.metrics import precision_score, recall_score

# 假設 y_pred 是模型預測的結果，y_test 是實際的結果
recall = recall_score(y_test, y_pred, average='macro')
print(f"Recall: {recall:.2f}")

"""
import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np

df = pd.read_csv('your_file.csv')

X = df.iloc[:, 39:]
y = df.iloc[:, :39]

X = X.to_numpy()
y = y.to_numpy()

X_train = X[:1000]
y_train = y[:1000]
X_test = X[1000:]
y_test = y[1000:]

model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=200)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')
"""

"""
import pandas as pd
from sklearn.linear_model import Lasso, LassoLars
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('OCE.csv')
X = df.iloc[:, 39:]
y = df.iloc[:, :39]

X = X.to_numpy()
y = y.to_numpy()

base_estimator = LassoLars(alpha=1.0, random_state=42)

# base_estimator = LassoLars(alpha=1.0, random_state=42)
model = MultiOutputRegressor(base_estimator)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(mse)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print(mae)

print(y_test)
print('---------------------')
print(y_pred)
"""

"""
import pandas as pd
df = pd.read_csv('OCE.csv')

X = df.iloc[:, 39:]
y = df.iloc[:, :39]

X = X.to_numpy()
y = y.to_numpy()

from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
base_estimator = MLPRegressor(hidden_layer_sizes=(100,),  # 隱藏層大小
                              activation='relu',  # 激活函數
                              solver='adam',  # 優化器
                              max_iter=200,  # 最大迭代次數
                              random_state=0,
                              alpha=1)
model = MultiOutputRegressor(base_estimator)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(mse)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print(mae)

print(y_test)
print('---------------------')
print(y_pred)
"""