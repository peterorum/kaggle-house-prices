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
from sklearn.preprocessing import LabelEncoder


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

    categorical_cols = [col for col in X_train_full.columns if X_train_full[col].dtype == 'object']

    numeric_cols.remove(target)

    X_train_full_numeric = X_train_full[numeric_cols]
    X_train_full_categorical = pd.DataFrame(X_train_full[categorical_cols])
    X_test_full_numeric = X_test_full[numeric_cols]
    X_test_full_categorical = pd.DataFrame(X_test_full[categorical_cols])

    # ----------------- missing values

    # impute missing values

    my_imputer = SimpleImputer()

    X_numeric = pd.DataFrame(my_imputer.fit_transform(X_train_full_numeric), columns=numeric_cols)
    X_test_numeric = pd.DataFrame(my_imputer.transform(X_test_full_numeric), columns=numeric_cols)

    # replace string nan with np.nan
    X_train_full_categorical.replace('nan', np.nan, inplace=True)
    X_test_full_categorical.replace('nan', np.nan, inplace=True)

    # replace missing categoricals with mode
    for col in categorical_cols:
        if X_train_full_categorical[col].isna().any() or X_test_full_categorical[col].isna().any():
            mode = X_train_full_categorical[col].mode()[0]

            X_train_full_categorical[col].fillna(mode, inplace=True)

            if col in X_test_full_categorical.columns:
                X_test_full_categorical[col].fillna(mode, inplace=True)

    # ---------------- categorical columns

    label_encoder = LabelEncoder()

    for col in categorical_cols:
        X_train_full_categorical[col] = label_encoder.fit_transform(X_train_full_categorical[col])
        X_test_full_categorical[col] = label_encoder.transform(X_test_full_categorical[col])

    # ---------------- join back together

    X = pd.concat([X_numeric, X_train_full_categorical], axis=1)

    X_test = pd.concat([X_test_numeric, X_test_full_categorical], axis=1)

    print(X)
    print(y)

    sys.exit(0)

    # ---------------- split into training & validation
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

    my_model = XGBRegressor(random_state=0)

    # --------------- fit the model to the training data & validate for score
    my_model.fit(X_train, y_train)
    preds_validate = my_model.predict(X_validate)
    score = np.sqrt(mean_squared_error(np.log(y_validate), np.log(preds_validate)))

    print(f'score: {score}')

    # --------------- generate test predictions on full set
    my_model.fit(X, y)

    preds_test = my_model.predict(X_test)

    # --------------- save predictions in format used for competition scoring
    output = pd.DataFrame({'Id': X_test.index,
                           target: preds_test})

    output.to_csv('submission.csv', index=False)


# -------- main

run()

print(f'Finished {((time() - start_time) / 60):.1f} mins\a')
