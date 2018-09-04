# linear regression
# lot area
# minimize score
# kaggle score 0.416

import os
import sys  # pylint: disable=unused-import
from time import time
import numpy as np  # pylint: disable=unused-import
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = ''  # if is_kaggle else '.zip'

# load data
train = pd.read_csv(f'../input/train.csv{zipext}')
test = pd.read_csv(f'../input/test.csv{zipext}')

#-------- main

start_time = time()

target = 'SalePrice'

train = train[['LotArea', target]]

x_train = train.drop(target, axis=1)
y_train = train[target]

linreg = LinearRegression()
linreg.fit(x_train, y_train)

# score = np.sqrt(mean_squared_error(train[target], train.predicted))
# print('score', score)

x_test = test[x_train.columns]
predicted = linreg.predict(x_test)

submission = pd.DataFrame({
    "ID": test.Id,
    "SalePrice": predicted
})

# print(test.head())
# print(test.describe())

submission.to_csv('submission.csv', index=False)

print('%.0f mins' % ((time() - start_time) / 60))
