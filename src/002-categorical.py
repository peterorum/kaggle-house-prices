# encode
# local score 0.132
# kaggle score 0.142
# minimize score

import csv
import os
import sys  # noqa
from time import time
from pprint import pprint  # noqa
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
np.set_printoptions(threshold=sys.maxsize)

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = ''  # if is_kaggle else '.zip'
train_file = 'train'  # if is_kaggle else 'sample'

start_time = time()
last_time = time()


def timer():
    global last_time

    print(f'{((time() - last_time) / 60):.1f}, {((time() - start_time) / 60):.1f} mins\n')  # noqa

    last_time = time()


def evaluate(train, test, unique_id, target):

    print('evaluate')

    model = XGBRegressor(random_state=0)

    x_train, x_test, y_train, y_test = train_test_split(train.drop(
        [unique_id, target], axis=1), train[target], test_size=0.2, random_state=1)

    model.fit(x_train, y_train)

    train_predictions = model.predict(x_test)
    train_score = np.sqrt(mean_squared_error(np.log(train_predictions), np.log(y_test)))

    test_predictions = model.predict(test[x_train.columns])

    timer()

    return test_predictions, train_score


# --- remove keys


def remove_keys(list, keys):

    result = [x for x in list if x not in keys]

    return result


# --- replace missing values


def replace_missing_values(train, test, unique_id, target):

    print(f'replace_missing_values')

    # use mean for numerics, mode for categorical

    numeric_cols = [col for col in train.columns
                    if (train[col].dtype == 'int64') | (train[col].dtype == 'float64')]

    numeric_cols = remove_keys(numeric_cols, [unique_id, target])

    categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
    categorical_cols = remove_keys(categorical_cols, [unique_id, target])

    # replace missing numericals with mean
    for col in numeric_cols:
        if train[col].isna().any() | test[col].isna().any():
            mean = train[col].mean()

            print(f'{col} mean {mean}')

            train[col].fillna(mean, inplace=True)

            if col in test.columns:
                test[col].fillna(mean, inplace=True)

    # convert to lowercase
    for col in categorical_cols:
        train[col] = train[col].apply(lambda x: str(x).lower())

        if col in test.columns:
            test[col] = test[col].apply(lambda x: str(x).lower())

    # replace string nan with np.nan
    train.replace('nan', np.nan, inplace=True)
    test.replace('nan', np.nan, inplace=True)

    # replace missing categoricals with mode
    for col in categorical_cols:
        if train[col].isna().any() or test[col].isna().any():
            mode = train[col].mode()[0]

            print(f'{col} mode {mean}')

            train[col].fillna(mode, inplace=True)

            if col in test.columns:
                test[col].fillna(mode, inplace=True)

    timer()

    return train, test

# --- categorical data


def encode_categorical_data(train, test, unique_id, target):

    print(f'get_categorical_data')

    train_targets = train[target]

    categorical_cols = [col for col in train.columns if train[col].dtype == 'object']

    if unique_id in categorical_cols:
        categorical_cols.remove(unique_id)

    # drop if too many values - usually a unique id column

    max_categories = train.shape[0] * 0.5

    too_many_value_categorical_cols = [col for col in categorical_cols
                                       if train[col].nunique() >= max_categories]

    if len(too_many_value_categorical_cols) > 0:
        print('dropping as too many categorical values', too_many_value_categorical_cols)

    categorical_cols = [i for i in categorical_cols if i not in too_many_value_categorical_cols]

    train = train.drop(too_many_value_categorical_cols, axis=1)
    test.drop([col for col in too_many_value_categorical_cols
               if col in test.columns], axis=1, inplace=True)

    # one-hot encode if not too many values

    max_ohe_categories = 15

    ohe_categorical_cols = [col for col in categorical_cols
                            if train[col].nunique() <= max_ohe_categories]

    categorical_cols = [i for i in categorical_cols if i not in ohe_categorical_cols]

    if len(ohe_categorical_cols) > 0:
        print('one-hot encode', ohe_categorical_cols)

        # one-hot encode & align to have same columns
        train = pd.get_dummies(train, columns=ohe_categorical_cols)
        test = pd.get_dummies(test, columns=ohe_categorical_cols)
        train, test = train.align(test, join='inner', axis=1)

        # restore after align
        train[target] = train_targets

    # possibly rank encode rather than ohe. see gstore.

    # label encode the remainder (convert to integer)

    label_encode_categorical_cols = categorical_cols

    print('label encode', label_encode_categorical_cols)

    for col in label_encode_categorical_cols:
        lbl = LabelEncoder()
        lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
        train[col] = lbl.transform(list(train[col].values.astype('str')))
        test[col] = lbl.transform(list(test[col].values.astype('str')))

    timer()

    return train, test

# --------------------- run


def run():

    unique_id = 'Id'
    target = 'SalePrice'

    # load data

    train = pd.read_csv(f'../input/{train_file}.csv{zipext}')
    test = pd.read_csv(f'../input/test.csv{zipext}')

    # ----------

    train, test = replace_missing_values(train, test, unique_id, target)

    train, test = encode_categorical_data(train, test, unique_id, target)

    # ----------

    test_predictions, train_score = evaluate(train, test, unique_id, target)

    print('score', train_score)

    test[target] = test_predictions

    predictions = test[[unique_id, target]]

    predictions.to_csv('submission.csv', index=False)


# -------- main

run()

print(f'Finished {((time() - start_time) / 60):.1f} mins\a')
