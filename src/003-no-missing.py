# encoded, no missing values 0.069
# kaggle score 0.1423
# minimize score

import os
import sys  # pylint: disable=unused-import
from time import time
import numpy as np  # pylint: disable=unused-import
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = ''  # if is_kaggle else '.zip'

# load data
train = pd.read_csv(f'../input/train.csv{zipext}')
test = pd.read_csv(f'../input/test.csv{zipext}')

#-------- main

start_time = time()

key = 'Id'
target = 'SalePrice'

# remove for combined analysis
train_targets = train[target]
train = train.drop(target, axis=1)

train['src'] = 'train'
test['src'] = 'test'
combined = train.append(test, ignore_index=True, sort=False)

# remove columns with missing values
combined = combined[[col for col in combined.columns if not combined[col].isna().any()]]

# encode categoricals so all numeric
categorical_cols = [col for col in combined.columns if combined[col].dtype == 'object' and col != 'src']

# drop if too many values
max_categories = combined.shape[0] * 0.5
many_value_categorical_cols = [col for col in categorical_cols if combined[col].nunique() >= max_categories]
few_value_categorical_cols = [col for col in categorical_cols if combined[col].nunique() < max_categories]

# encode
combined = pd.get_dummies(combined, columns=few_value_categorical_cols)

combined = combined.drop(many_value_categorical_cols, axis=1)

# reformat col names
combined.columns = [col.replace(' ', '_') for col in combined.columns.tolist()]

# separate
train = combined[combined.src == 'train'].drop('src', axis=1)
test = combined[combined.src == 'test'].drop('src', axis=1)

x_train = train
y_train = train_targets
x_test = test[x_train.columns]

model = lgb.LGBMRegressor()

model.fit(x_train, y_train)

train['predicted'] = model.predict(x_train)

score = np.sqrt(mean_squared_error(np.log(train_targets), np.log(train.predicted)))
print('score', score)

predicted = model.predict(x_test)

submission = pd.DataFrame({
    "ID": test.Id,
    "SalePrice": predicted
})

# print(test.head())
# print(test.describe())

submission.to_csv('submission.csv', index=False)

print('%.0f mins\a' % ((time() - start_time) / 60))
