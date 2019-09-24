# train/test split
# local score  0.132
# kaggle score 0.141
# minimize score

import pandas as pd
import numpy as np
from time import time
import sys  # noqa

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

start_time = time()
last_time = time()


def timer():
    global last_time

    print(f'{((time() - last_time) / 60):.1f} mins\n')  # noqa

    last_time = time()


# --------------------- run


def run():

    target = 'SalePrice'

    # Read the data
    X_train_full = pd.read_csv('../input/train.csv', index_col='Id')
    X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

    # Obtain target and predictors
    y = X_train_full[target]

    numeric_cols = [col for col in X_train_full.columns
                    if (X_train_full[col].dtype == 'int64') | (X_train_full[col].dtype == 'float64')]

    features = numeric_cols
    features.remove(target)

    X = X_train_full[features].copy()
    X_test = X_test_full[features].copy()

    # missing values
    cols_with_missing = [col for col in features
                         if X[col].isnull().any()]

    # impute missing values

    column_names = X.columns

    my_imputer = SimpleImputer()
    X = pd.DataFrame(my_imputer.fit_transform(X))
    X_test = pd.DataFrame(my_imputer.transform(X_test))

    # restore column names
    X.columns = column_names
    X_test.columns = column_names

    # split into training & validation
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

    my_model = XGBRegressor(random_state=0)

    # Fit the model to the training data & validate for score
    my_model.fit(X_train, y_train)
    preds_validate = my_model.predict(X_validate)
    score = np.sqrt(mean_squared_error(np.log(y_validate), np.log(preds_validate)))

    print(f'score: {score}')

    # Generate test predictions on full set
    my_model.fit(X, y)

    preds_test = my_model.predict(X_test)

    # Save predictions in format used for competition scoring
    output = pd.DataFrame({'Id': X_test.index,
                           target: preds_test})

    output.to_csv('submission.csv', index=False)


# -------- main

run()

print(f'Finished {((time() - start_time) / 60):.1f} mins\a')
