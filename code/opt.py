import optuna
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold
df = pd.read_csv('OCE.csv')

X = df.iloc[:, 39:]
y = df.iloc[:, :39]

X = X.to_numpy()
y = y.to_numpy()

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 400)
    max_depth = trial.suggest_int('max_depth', 2, 50)
    alpha=trial.suggest_loguniform('alpha', 1e-6, 1)
    learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

    base_estimator = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                max_depth=max_depth, random_state=42,alpha=alpha)
    

    model = MultiOutputRegressor(base_estimator)

    kfold = KFold(n_splits=3)
    i = 0
    s = 0
    for train_index, test_index in kfold.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        model.fit(X_train, y_train)
        s = max(s,model.score(X_test,y_test,sample_weight=None))
        i = i + 1
    return s

study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
study.optimize(objective, n_trials=100)

print(study.best_params)