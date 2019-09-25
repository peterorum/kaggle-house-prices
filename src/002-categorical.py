# deal with categories
# local score 0.126
# kaggle score 0.1363
# minimize score

import csv
import os
import sys  # noqa
import operator
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

# run the model


def evaluate(train, test, unique_id, target):

    print('evaluate')

    model = XGBRegressor(random_state=0)

    x_train, x_validate, y_train, y_validate = train_test_split(train.drop(
        [unique_id, target], axis=1), train[target], test_size=0.2, random_state=1)

    model.fit(x_train, y_train)

    train_predictions = model.predict(x_validate)
    train_score = np.sqrt(mean_squared_error(np.log(train_predictions), np.log(y_validate)))

    test_predictions = model.predict(test[x_train.columns])

    timer()

    return test_predictions, train_score


# get top 10 important features


def get_important_features(train, unique_id, target):

    print('get_important_features')

    model = XGBRegressor(random_state=0)

    model.fit(train.drop([unique_id, target], axis=1), train[target])

    important_features = model.get_booster().get_score(importance_type='weight')
    important_features = sorted(important_features.items(), key=operator.itemgetter(1), reverse=True)
    important_features = [x[0] for x in important_features[0:10]]

    timer()

    return important_features


# --- remove keys


def remove_keys(list, keys):

    result = [x for x in list if x not in keys]

    return result


# clear empty numeric values that should not get a mean


def clear_missing_values(train, test, columns, value):

    for col in columns:
        train[col] = train[col].fillna(value)
        test[col] = test[col].fillna(value)

    return train, test


# convert numeric columns which are actually just categories


def convert_numeric_categories(train, test, columns):

    for col in columns:
        train[col] = train[col].apply(str)
        test[col] = test[col].apply(str)

    return train, test

# --- replace missing values


def replace_missing_values(train, test, unique_id, target):

    print(f'replace_missing_values')

    # print(f'columns with missing data {train.columns[train.isna().any()]}')

    # use mean for numerics, mode for categoric

    numeric_cols = [col for col in train.columns
                    if (train[col].dtype == 'int64') | (train[col].dtype == 'float64')]

    numeric_cols = remove_keys(numeric_cols, [unique_id, target])

    categoric_cols = [col for col in train.columns if train[col].dtype == 'object']
    categoric_cols = remove_keys(categoric_cols, [unique_id, target])

    # replace missing numericals with mean
    for col in numeric_cols:
        if train[col].isna().any() | test[col].isna().any():
            mean = train[col].mean()

            # print(f'{col} mean {mean}')

            train[col].fillna(mean, inplace=True)

            if col in test.columns:
                test[col].fillna(mean, inplace=True)

    # convert to lowercase
    for col in categoric_cols:
        train[col] = train[col].apply(lambda x: str(x).lower())

        if col in test.columns:
            test[col] = test[col].apply(lambda x: str(x).lower())

    # replace string nan with np.nan
    train.replace('nan', np.nan, inplace=True)
    test.replace('nan', np.nan, inplace=True)

    # replace missing categorics with mode
    for col in categoric_cols:
        if train[col].isna().any() or test[col].isna().any():
            mode = train[col].mode()[0]

            # print(f'{col} mode {mode}')

            train[col].fillna(mode, inplace=True)

            if col in test.columns:
                test[col].fillna(mode, inplace=True)

    timer()

    return train, test

# --- categoric data


def encode_categoric_data(train, test, unique_id, target):

    print(f'get_categoric_data')

    train_targets = train[target]

    categoric_cols = [col for col in train.columns if train[col].dtype == 'object']

    if unique_id in categoric_cols:
        categoric_cols.remove(unique_id)

    # drop if too many values - usually a unique id column

    max_categories = train.shape[0] * 0.5

    too_many_value_categoric_cols = [col for col in categoric_cols
                                     if train[col].nunique() >= max_categories]

    if len(too_many_value_categoric_cols) > 0:
        print('dropping as too many categoric values', too_many_value_categoric_cols)

    categoric_cols = [i for i in categoric_cols if i not in too_many_value_categoric_cols]

    train = train.drop(too_many_value_categoric_cols, axis=1)
    test.drop([col for col in too_many_value_categoric_cols
               if col in test.columns], axis=1, inplace=True)

    # one-hot encode if not too many values

    max_ohe_categories = 15

    ohe_categoric_cols = [col for col in categoric_cols
                          if train[col].nunique() <= max_ohe_categories]

    categoric_cols = [i for i in categoric_cols if i not in ohe_categoric_cols]

    if len(ohe_categoric_cols) > 0:
        # print('one-hot encode', ohe_categoric_cols)

        # one-hot encode & align to have same columns
        train = pd.get_dummies(train, columns=ohe_categoric_cols)
        test = pd.get_dummies(test, columns=ohe_categoric_cols)
        train, test = train.align(test, join='inner', axis=1)

        # restore after align
        train[target] = train_targets

    # possibly rank encode rather than ohe. see gstore.

    # label encode the remainder (convert to integer)

    label_encode_categoric_cols = categoric_cols

    # print('label encode', label_encode_categoric_cols)

    for col in label_encode_categoric_cols:
        lbl = LabelEncoder()
        lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
        train[col] = lbl.transform(list(train[col].values.astype('str')))
        test[col] = lbl.transform(list(test[col].values.astype('str')))

    timer()

    return train, test


# custom features

def add_custom_features(train, test, unique_id, target):

    print(f'add_custom_features')

    for df in [train, test]:
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

        df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] +
                                   df['1stFlrSF'] + df['2ndFlrSF'])

        df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) +
                                 df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

    timer()

    return train, test


# --------------------- run


def run():

    unique_id = 'Id'
    target = 'SalePrice'

    # load data

    train = pd.read_csv(f'../input/{train_file}.csv{zipext}')
    test = pd.read_csv(f'../input/test.csv{zipext}')

    original_columns = train.columns.tolist()

    # ----------

    train, test = clear_missing_values(train, test, ['GarageYrBlt', 'GarageArea', 'GarageCars'], 0)
    train, test = clear_missing_values(
        train, test, ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'FireplaceQu', 'Alley', 'BsmtQual',
                      'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'PoolQC', 'Fence', 'MiscFeature'], 'NA')

    train, test = convert_numeric_categories(train, test, ['MSSubClass'])

    train, test = replace_missing_values(train, test, unique_id, target)

    train, test = add_custom_features(train, test, unique_id, target)

    train, test = encode_categoric_data(train, test, unique_id, target)

    important_features = get_important_features(train, unique_id, target)

    # ----------

    test_predictions, train_score = evaluate(train, test, unique_id, target)

    print('score', train_score)

    test[target] = test_predictions

    predictions = test[[unique_id, target]]

    predictions.to_csv('submission.csv', index=False)


# -------- main

run()

print(f'Finished {((time() - start_time) / 60):.1f} mins\a')
