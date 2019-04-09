# feature generation & selection
# sample
# full
# kaggle 0.14481
# minimize score

import os
import json
import sys  # pylint: disable=unused-import
from time import time
import csv
from pprint import pprint  # pylint: disable=unused-import
from timeit import default_timer as timer
import lightgbm as lgb
import numpy as np
from hyperopt import STATUS_OK, fmin, hp, tpe, Trials
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures

pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)

is_kaggle = os.environ['HOME'] == '/tmp'

use_sample = False if is_kaggle else True

# hyperopt
optimize = False
max_evals = 200 if is_kaggle else 100
results_file = 'optimize.csv'
iteration = 0
best_score = sys.float_info.max
trials = Trials()

optimized_params = {
    'class_weight': None,
    'colsample_bytree': 1.0,
    'learning_rate': 0.1,
    'min_child_samples': 20,
    'num_leaves': 31,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'subsample_for_bin': 20000
}

zipext = '' if is_kaggle else '.zip'

# params
n_folds = 10
stop_rounds = 100
verbose_eval = -1  # 500

# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields


def evaluate(params):

    # defaults
    # LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
    #        learning_rate=0.1, max_depth=-1, min_child_samples=20,
    #        min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
    #        n_jobs=-1, nthread=4, num_leaves=31, objective=None,
    #        random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
    #        subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    params['num_leaves'] = int(params['num_leaves'])
    params['min_child_samples'] = int(params['min_child_samples'])
    params['subsample_for_bin'] = int(params['subsample_for_bin'])

    if params['class_weight'] == 0:
        params['class_weight'] = None

    lgb_model = lgb.LGBMRegressor(nthread=4, n_jobs=-1, verbose=-1)
    lgb_model.set_params(**params)

    lr_model = LinearRegression()

    test_predictions = np.zeros(test.shape[0])
    best_score = 0

    do_ensemble = False

    ensemble_count = 2 if do_ensemble else 1

    for fold_n, (train_index, test_index) in enumerate(folds.split(x_train)):
        X_train, X_valid = x_train.iloc[train_index], x_train.iloc[test_index]
        Y_train, Y_valid = y_train.iloc[train_index], y_train.iloc[test_index]

        # lgb
        lgb_model.fit(X_train, Y_train,
                      eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
                      eval_metric='rmse',
                      verbose=False, early_stopping_rounds=stop_rounds)

        # pprint(dir(lgb_model))
        best_score += lgb_model.best_score_['valid_1']['rmse']

        lgb_test_prediction = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration_)
        test_predictions += lgb_test_prediction

        if do_ensemble:
            # linear regression
            lr_model.fit(X_train, Y_train)
            train_prediction = lr_model.predict(X_train)
            best_score += np.sqrt(mean_squared_error(train_prediction, Y_train))

            lr_test_prediction = lr_model.predict(x_test)
            test_predictions += lr_test_prediction

    test_predictions /= (n_folds * ensemble_count)
    best_score /= (n_folds * ensemble_count)

    return test_predictions, best_score


def objective(params):

    global iteration, best_score
    iteration += 1

    start = timer()

    _, score = evaluate(params)

    run_time = timer() - start

    # save results
    of_connection = open(results_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([iteration, score, run_time, params])
    of_connection.close()

    # save trials for resumption
    # with open('trials.json', 'w') as f:
    #     # might be trials_dict to be saved
    #     f.write(json.dumps(trials))

    best_score = min(best_score, score)

    print(f'iteration {iteration}, score {score}, best {best_score}, timer {run_time}')

    # score must be to minimize

    return {'loss': score, 'params': params, 'iteration': iteration,
            'train_time': run_time, 'status': STATUS_OK}


# polynomial features


def get_polynomial_features(train, test, target):

    # Make a new dataframe for polynomial features

    numeric_cols = [col for col in train.columns
                    if (col != target) & (col != unique_id) & ((train[col].dtype == 'int64') | (train[col].dtype == 'float64'))]

    poly_features = train[numeric_cols]
    poly_features_test = test[numeric_cols]

    poly_target = train_targets

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree=2)

    # Train the polynomial features
    poly_transformer.fit(poly_features)

    # Transform the features
    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)

    # print('\nPolynomial Features shape: ', poly_features.shape)
    # print(poly_transformer.get_feature_names(input_features=numeric_cols))

    # Create a dataframe of the features
    poly_features = pd.DataFrame(poly_features,
                                 columns=poly_transformer.get_feature_names(numeric_cols))

    # Add in the target
    poly_features[target] = poly_target

    # Find the correlations with the target
    poly_corrs = poly_features.corr()[target].sort_values()

    # Display most negative and most positive
    # print(poly_corrs.head(10))
    # print(poly_corrs.tail(5))

    # Put test features into dataframe
    poly_features_test = pd.DataFrame(poly_features_test,
                                      columns=poly_transformer.get_feature_names(numeric_cols))

    # Merge polynomial features into training dataframe

    poly_features[unique_id] = train[unique_id]

    train_poly = train.merge(poly_features, on=unique_id, how='left')

    # print('\nPolynomial Features shape: ', poly_features.shape, train.shape, train_poly.shape, )
    # print('\nPolynomial Features shape: ', poly_features.describe(), train_poly.describe(), train.describe())

    # Merge polynomial features into testing dataframe
    poly_features_test[unique_id] = test[unique_id]
    test_poly = test.merge(poly_features_test, on=unique_id, how='left')

    # Align the dataframes
    train_poly, test_poly = train_poly.align(test_poly, join='inner', axis=1)

    # Print out the new shapes
    # print('Training data with polynomial features shape: ', train_poly.shape)
    # print('Testing data with polynomial features shape:  ', test_poly.shape)

    # train_poly, test_poly = get_collinear_features(train_poly, test_poly, target)

    return train_poly, test_poly

# arithmetic features


def get_arithmetic_features(train, test, target):

    numeric_cols = [col for col in train.columns
                    if (train[col].dtype == 'int64') | (train[col].dtype == 'float64')]

    numeric_cols.remove(unique_id)

    for i1 in range(0, len(numeric_cols)):
        col1 = numeric_cols[i1]
        for i2 in range(i1 + 1, len(numeric_cols)):
            col2 = numeric_cols[i2]

            # train[f'{col1} by {col2}'] = train[col1] * train[col2]
            # test[f'{col1} by {col2}'] = test[col1] * test[col2]

            # train[f'{col1} plus {col2}'] = train[col1] + train[col2]
            # test[f'{col1} plus {col2}'] = test[col1] + test[col2]

            train[f'{col1} minus {col2}'] = train[col1] - train[col2]
            test[f'{col1} minus {col2}'] = test[col1] - test[col2]

            # if not (train[col2] == 0).any():
            #     train[f'{col1} on {col2}'] = train[col1] / train[col2]
            #     test[f'{col1} on {col2}'] = test[col1] / test[col2]
            # elif not (train[col1] == 0).any():
            #     train[f'{col2} on {col1}'] = train[col2] / train[col1]
            #     test[f'{col2} on {col1}'] = test[col2] / test[col1]

    train, test = get_collinear_features(train, test, target)

    return train, test


def get_collinear_features(train, test, target):
    corrs = train.corr()
    upper = corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(np.bool))

    threshold = 0.8

    # Select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    train = train.drop(columns=to_drop)
    test = test.drop(columns=to_drop)

    return train, test


def get_missing_values(train, test, target):

    threshold = 0.75

    train_missing = (train.isnull().sum() / len(train)).sort_values(ascending=False)
    test_missing = (test.isnull().sum() / len(test)).sort_values(ascending=False)
    # print(train_missing.head())
    # print(test_missing.head())

    # Identify missing values above threshold
    train_missing = train_missing.index[train_missing > threshold]
    test_missing = test_missing.index[test_missing > threshold]

    all_missing = list(set(set(train_missing) | set(test_missing)))
    # print(f'There are {len(all_missing)} columns with more than {threshold}%% missing values')

    train = train.drop(columns=all_missing)
    test = test.drop(columns=all_missing)

    train, test = train.align(test, join='inner', axis=1)

    return train, test


def get_feature_importance(train, test, target):

    model = lgb.LGBMRegressor(nthread=4, n_jobs=-1, verbose=-1)
    model.set_params(**optimized_params)

    x_train = train.drop(target_key, axis=1)

    if unique_id in x_train.columns:
        x_train = train.drop(target_key, axis=1)

    # Initialize an empty array to hold feature importances
    feature_importances = np.zeros(x_train.shape[1])

    # Fit the model twice to avoid overfitting
    for i in range(2):

        # Split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(x_train, train_targets,
                                                                            test_size=0.25, random_state=i)

        # Train using early stopping
        model.fit(train_features, train_y, early_stopping_rounds=100,
                  eval_set=[(valid_features, valid_y)],
                  eval_metric='rmse', verbose=False)

        # Record the feature importances
        feature_importances += model.feature_importances_

    # Make sure to average feature importances!
    feature_importances = feature_importances / 2

    feature_importances = pd.DataFrame(
        {'feature': list(x_train.columns), 'importance': feature_importances}).sort_values('importance', ascending=False)

    # Sort features according to importance
    feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    feature_importances['importance_normalized'] = feature_importances['importance'] / feature_importances['importance'].sum()
    feature_importances['cumulative_importance'] = np.cumsum(feature_importances['importance_normalized'])

    # print(feature_importances)

    # Find the features with minimal importance
    # unimportant_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])

    # Threshold for cumulative importance
    threshold = 0.99

    # Extract the features to drop

    features_to_drop = list(feature_importances[feature_importances[
        'cumulative_importance'] > threshold]['feature'])

    # print(f'There are {len(features_to_drop)} features under {threshold} importance')
    # print(features_to_drop)

    train = train.drop(features_to_drop, axis=1)
    test = test.drop(features_to_drop, axis=1)

    return train, test


def get_feature_selection(train, test, target):
    # remove collinear variables
    train, test = get_collinear_features(train, test, target)
    print(f'collinear, cols {len(train.columns)}, {((time() - start_time) / 60):.0f} mins')

    all_numeric_cols = [col for col in train.columns
                        if (col != unique_id) & (train[col].dtype == 'int64') | (train[col].dtype == 'float64')]

    # feature selection via variance
    train_numeric = train[all_numeric_cols].fillna(0)
    select_features = VarianceThreshold(threshold=0.2)
    select_features.fit(train_numeric)
    numeric_cols = train_numeric.columns[select_features.get_support(indices=True)].tolist()

    # remove cols without variance
    for col in all_numeric_cols:
        if col not in numeric_cols:
            train.drop(col, axis=1, inplace=True)

            if col in test.columns:
                test.drop(col, axis=1, inplace=True)

    print(f'variance, cols {len(train.columns)}, {((time() - start_time) / 60):.0f} mins')

    # determine important featuers
    train, test = get_feature_importance(train, test, target)
    print(f'importance, cols {len(train.columns)}, {((time() - start_time) / 60):.0f} mins')

    return train, test

# -------- main


start_time = time()

unique_id = 'unique_id'
target_key = 'Id'
target = 'SalePrice'

# load data

if use_sample:
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
else:
    train = pd.read_csv(f'../input/train.csv{zipext}')
    test = pd.read_csv(f'../input/test.csv{zipext}')

# add unique id
train[unique_id] = range(1, len(train.index) + 1)
test[unique_id] = range(1, len(test.index) + 1)

print(f'load data, cols {len(train.columns)}, {((time() - start_time) / 60):.0f} mins')

# specific conversions required
int_cols = []

float_cols = []

for col in int_cols:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)

for col in float_cols:
    train[col] = train[col].astype(float)

    if col != target:
        test[col] = test[col].astype(float)

train[target] = train[target].fillna(0)

train_targets = train[target]

# remove columns with many missing values
train, test = get_missing_values(train, test, target)
print(f'missing, cols {len(train.columns)}, {((time() - start_time) / 60):.0f} mins')

#----------

all_numeric_cols = [col for col in train.columns
                    if (train[col].dtype == 'int64') | (train[col].dtype == 'float64')]
all_numeric_cols.remove(unique_id)

categorical_cols = [col for col in train.columns if train[col].dtype == 'object']

if target_key in categorical_cols:
    categorical_cols.remove(target_key)

# replace missing numericals with mean
for col in all_numeric_cols:
    if train[col].isna().any() | test[col].isna().any():
        mean = train[col].mean()

        train[col].fillna(mean, inplace=True)

        if col in test.columns:
            test[col].fillna(mean, inplace=True)

# replace missing categoricals with mode
for col in categorical_cols:
    if train[col].isna().any():
        mode = train[col].mode()[0]

        train[col].fillna(mode, inplace=True)

        if col in test.columns:
            test[col].fillna(mode, inplace=True)

# arithmetic features
train, test = get_arithmetic_features(train, test, target)
print(f'arithmetic, cols {len(train.columns)}, {((time() - start_time) / 60):.0f} mins')

# polynomial features
train, test = get_polynomial_features(train, test, target)
print(f'polynomial, cols {len(train.columns)}, {((time() - start_time) / 60):.0f} mins')

# encode categoricals so all numeric

# drop if too many values, encode if few
# one-hot-encode up to 100 categories, else label encode
max_categories = train.shape[0] * 0.5
max_ohe_categories = 100  # use 0 to disable ohe

too_many_value_categorical_cols = [col for col in categorical_cols
                                   if train[col].nunique() >= max_categories]

# drop if too many value s- usually a unique id column
train = train.drop(too_many_value_categorical_cols, axis=1)
test.drop([col for col in too_many_value_categorical_cols
           if col in test.columns], axis=1, inplace=True)

categorical_cols = np.setdiff1d(categorical_cols, too_many_value_categorical_cols)

ohe_categorical_cols = [col for col in categorical_cols
                        if train[col].nunique() <= max_ohe_categories]

label_encode_categorical_cols = [col for col in categorical_cols
                                 if (train[col].nunique() > max_ohe_categories)
                                 & (train[col].nunique() < max_categories)]

# label encode
for col in label_encode_categorical_cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))

# one-hot encode & align to have same columns
train = pd.get_dummies(train, columns=ohe_categorical_cols)
test = pd.get_dummies(test, columns=ohe_categorical_cols)
train, test = train.align(test, join='inner', axis=1)

print(f'encoded, cols {len(train.columns)}, {((time() - start_time) / 60):.0f} mins')

#-- feature selection (reduction)

train, test = get_feature_selection(train, test, target)

# -------- train

# reformat col names
train.columns = [col.replace(' ', '_') for col in train.columns.tolist()]
test.columns = [col.replace(' ', '_') for col in test.columns.tolist()]

# kfolds
folds = KFold(n_splits=n_folds, shuffle=True, random_state=1)

x_train = train.drop(target_key, axis=1)

if unique_id in x_train.columns:
    x_train = train.drop(target_key, axis=1)


y_train = train_targets
x_test = test[x_train.columns]

# pprint(list(x_train.columns))

if optimize:
    # define the search space
    space = {
        'class_weight': hp.choice('class_weight', [None, 'balanced']),
        'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
        'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
    }

    of_connection = open(results_file, 'w')
    writer = csv.writer(of_connection)
    writer.writerow(['iteration', 'score', 'run_time', 'params'])
    of_connection.close()

    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=max_evals, trials=trials)

    print('best', best)

    # pprint(trials.results)
    trials_dict = sorted(trials.results, key=lambda x: x['loss'])
    print(f'score {trials_dict[:1][0]["loss"]}')

else:
    test_predictions, score = evaluate(optimized_params)

    print('score', score)

    test['predicted'] = test_predictions

    submission = pd.DataFrame({
        "ID": test.Id,
        "SalePrice": test.predicted
    })

    submission.to_csv('submission.csv', index=False)


print(f'{((time() - start_time) / 60):.0f} mins\a')
