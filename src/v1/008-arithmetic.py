# feature importance
# local score 0.05299
# kaggle score .13349
# minimize score

import os
import sys  # noqa
from time import time
from pprint import pprint  # noqa
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

    print(f'{((time() - last_time) / 60):.1f} mins\n')  # noqa

    last_time = time()


def evaluate(train, test, unique_id, target):

    print('evaluate')

    lgb_model = lgb.LGBMRegressor(nthread=4, n_jobs=-1, verbose=-1, metric='rmse')

    x_train = train.drop([target, unique_id], axis=1)
    y_train = train[target]

    x_test = test[x_train.columns]

    lgb_model.fit(x_train, y_train)

    train_predictions = lgb_model.predict(x_train)
    test_predictions = lgb_model.predict(x_test)

    train_score = np.sqrt(mean_squared_error(np.log(train_predictions), np.log(y_train)))

    timer()

    return test_predictions, train_score

# --- missing values


def get_many_missing_values(train, test, unique_id, target):

    print(f'get_many_missing_values')

    train_targets = train[target]

    threshold = 0.75

    train_missing = (train.isnull().sum() / len(train)).sort_values(ascending=False)
    test_missing = (test.isnull().sum() / len(test)).sort_values(ascending=False)

    # identify missing values above threshold
    train_missing = train_missing.index[train_missing > threshold]
    test_missing = test_missing.index[test_missing > threshold]

    all_missing = list(set(set(train_missing) | set(test_missing)))

    if len(all_missing) > 0:
        print(f'columns with more than {threshold}% missing values')
        pprint(all_missing)

        train = train.drop(columns=all_missing, axis=1)
        test = test.drop(columns=all_missing, axis=1)

        train, test = train.align(test, join='inner', axis=1)

        # restore after align
        train[target] = train_targets

    timer()

    return train, test

# --- remove keys


def remove_keys(list, keys):

    result = [x for x in list if x not in keys]

    return result

# --- replace missing values


def replace_missing_values(train, test, unique_id, target):

    print(f'replace_missing_values')

    numeric_cols = [col for col in train.columns
                    if (train[col].dtype == 'int64') | (train[col].dtype == 'float64')]

    numeric_cols = remove_keys(numeric_cols, [unique_id, target])

    categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
    categorical_cols = remove_keys(categorical_cols, [unique_id, target])

    # replace missing numericals with mean
    for col in numeric_cols:
        if train[col].isna().any() | test[col].isna().any():
            mean = train[col].mean()

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

            train[col].fillna(mode, inplace=True)

            if col in test.columns:
                test[col].fillna(mode, inplace=True)

    timer()

    return train, test

# --- column differences


def get_column_differences(train, test, unique_id, target):

    print(f'get_column_differences')

    train_without_target = train.drop(target, axis=1)

    not_in_test = train_without_target.columns.difference(test.columns)
    not_in_train = test.columns.difference(train_without_target.columns)

    if len(not_in_test) > 0:
        print(f'In train but not test\n{not_in_test}')

    if len(not_in_train) > 0:
        print(f'In test but not train\n{not_in_train}')

    timer()

    return train, test


# --- categorical data


def get_categorical_data(train, test, unique_id, target):

    print(f'get_categorical_data')

    train_targets = train[target]

    categorical_cols = [col for col in train.columns if train[col].dtype == 'object']

    if unique_id in categorical_cols:
        categorical_cols.remove(unique_id)

    max_categories = train.shape[0] * 0.5

    too_many_value_categorical_cols = [col for col in categorical_cols
                                       if train[col].nunique() >= max_categories]

    if len(too_many_value_categorical_cols) > 0:
        print('too many categorical values', too_many_value_categorical_cols)

    # drop if too many values - usually a unique id column

    categorical_cols = [i for i in categorical_cols if i not in too_many_value_categorical_cols]

    train = train.drop(too_many_value_categorical_cols, axis=1)
    test.drop([col for col in too_many_value_categorical_cols
               if col in test.columns], axis=1, inplace=True)

    # one-hot encode if not too many values

    max_ohe_categories = 10

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

    # label encode (convert to integer)

    label_encode_categorical_cols = categorical_cols

    print('label encode', label_encode_categorical_cols)

    for col in label_encode_categorical_cols:
        lbl = LabelEncoder()
        lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
        train[col] = lbl.transform(list(train[col].values.astype('str')))
        test[col] = lbl.transform(list(test[col].values.astype('str')))

    timer()

    return train, test

# --- feature selection


def get_feature_selection(train, test, unique_id, target):

    print(f'get_feature_selection')

    all_numeric_cols = [col for col in train.columns
                        if (train[col].dtype == 'int64') | (train[col].dtype == 'float64')]

    if unique_id in all_numeric_cols:
        all_numeric_cols.remove(unique_id)

    if target in all_numeric_cols:
        all_numeric_cols.remove(target)

    # feature selection via variance
    train_numeric = train[all_numeric_cols].fillna(0)
    select_features = VarianceThreshold(threshold=0.2)
    select_features.fit(train_numeric)
    numeric_cols = train_numeric.columns[select_features.get_support(indices=True)].tolist()

    # remove cols without variance
    for col in all_numeric_cols:
        if col not in numeric_cols:
            print(f'variance drop {col}')
            train.drop(col, axis=1, inplace=True)

            if col in test.columns:
                test.drop(col, axis=1, inplace=True)

    timer()

    return train, test

# --- feature importance


def get_feature_importance(train, test, unique_id, target):

    print(f'get_feature_importance')

    model = lgb.LGBMRegressor(nthread=4, n_jobs=-1, verbose=-1)

    x_train = train.drop([unique_id, target], axis=1)

    # initialize an empty array to hold feature importances
    feature_importances = np.zeros(x_train.shape[1])

    # fit the model twice to avoid overfitting
    for i in range(2):

        # split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(x_train, train[target],
                                                                            test_size=0.25, random_state=i)

        # train using early stopping
        model.fit(train_features, train_y, early_stopping_rounds=100,
                  eval_set=[(valid_features, valid_y)],
                  eval_metric='rmse', verbose=False)

        # record the feature importances
        feature_importances += model.feature_importances_

    # average feature importances!
    feature_importances = feature_importances / 2

    feature_importances = pd.DataFrame(
        {'feature': list(x_train.columns), 'importance': feature_importances}).sort_values('importance', ascending=False)

    # sort features according to importance
    feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index()

    most_important_features = feature_importances[0:10]['feature'].tolist()

    # normalize the feature importances to add up to one
    feature_importances['importance_normalized'] = feature_importances['importance'] / feature_importances['importance'].sum()
    feature_importances['cumulative_importance'] = np.cumsum(feature_importances['importance_normalized'])

    # find the features with minimal importance
    # unimportant_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])

    # Threshold for cumulative importance
    threshold = 0.9996

    # extract the features to drop

    features_to_drop = list(feature_importances[feature_importances[
        'cumulative_importance'] > threshold]['feature'])

    if len(features_to_drop) > 0:
        print(feature_importances)

        print(f'features to drop, under {threshold} importance:')
        pprint(features_to_drop)

        train = train.drop(features_to_drop, axis=1)
        test = test.drop(features_to_drop, axis=1)

    timer()

    return train, test, most_important_features

# --- remove collinear features


def get_collinear_features(train, test, unique_id, target):

    print('get_collinear_features')

    corrs = train.corr()
    upper = corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(np.bool))

    threshold = 0.8

    # select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold) and column not in [unique_id, target]]

    if len(to_drop) > 0:
        print('collinear drop')
        pprint(to_drop)

        train = train.drop(columns=to_drop, axis=1)
        test = test.drop(columns=to_drop, axis=1)

    timer()

    return train, test


# arithmetic features


def get_arithmetic_features(train, test, unique_id, target, cols, source_cols):

    print('get_arithmetic_features')

    # just choose from original columns, not encodeds

    numeric_cols = [col for col in cols
                    if (col in source_cols) & (train[col].dtype == 'int64') | (train[col].dtype == 'float64')]

    numeric_cols = remove_keys(numeric_cols, [unique_id, target])

    for i1 in range(0, len(numeric_cols)):
        col1 = numeric_cols[i1]

        # powers
        train[f'{col1} squared'] = train[col1] ** 2
        test[f'{col1} squared'] = test[col1] ** 2
        train[f'{col1} cubed'] = train[col1] ** 3
        test[f'{col1} cubed'] = test[col1] ** 3
        train[f'{col1}^4'] = train[col1] ** 4
        test[f'{col1}^4'] = test[col1] ** 4

        for i2 in range(i1 + 1, len(numeric_cols)):
            col2 = numeric_cols[i2]

            train[f'{col1} by {col2}'] = train[col1] * train[col2]
            test[f'{col1} by {col2}'] = test[col1] * test[col2]

            train[f'{col1} plus {col2}'] = train[col1] + train[col2]
            test[f'{col1} plus {col2}'] = test[col1] + test[col2]

            train[f'{col1} minus {col2}'] = train[col1] - train[col2]
            test[f'{col1} minus {col2}'] = test[col1] - test[col2]

            if not (train[col2] == 0).any():
                train[f'{col1} on {col2}'] = train[col1] / train[col2]
                test[f'{col1} on {col2}'] = test[col1] / test[col2]
            elif not (train[col1] == 0).any():
                train[f'{col2} on {col1}'] = train[col2] / train[col1]
                test[f'{col2} on {col1}'] = test[col2] / test[col1]

    timer()

    return train, test


# custom features

def get_custom_features(train, test, unique_id, target):

    print(f'get_custom_features')
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

    train, test = get_many_missing_values(train, test, unique_id, target)

    train, test = replace_missing_values(train, test, unique_id, target)

    train, test = get_column_differences(train, test, unique_id, target)

    train, test = get_custom_features(train, test, unique_id, target)

    train, test = get_categorical_data(train, test, unique_id, target)

    train, test, most_important_cols = get_feature_importance(train, test, unique_id, target)

    train, test = get_arithmetic_features(train, test, unique_id, target, most_important_cols, original_columns)

    train, test = get_collinear_features(train, test, unique_id, target)

    train, test = get_feature_selection(train, test, unique_id, target)

    train, test, _ = get_feature_importance(train, test, unique_id, target)

    # ----------

    test_predictions, train_score = evaluate(train, test, unique_id, target)

    print('score', train_score)

    test[target] = test_predictions

    predictions = test[[unique_id, target]]

    predictions.to_csv('submission.csv', index=False)


# -------- main

run()

print(f'Finished {((time() - start_time) / 60):.1f} mins\a')
