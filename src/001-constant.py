# baseline: constant price 200000
# kaggle score 0.1886

import sys  # pylint: disable=unused-import
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from time import time

import os

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = ''  # if is_kaggle else '.zip'

# load data
train = pd.read_csv(f'../input/train.csv{zipext}')
test = pd.read_csv(f'../input/test.csv{zipext}')

#-------- main

start_time = time()

target = 'SalePrice'

result = 200000

train['predicted'] = result

score = np.sqrt(mean_squared_error(train[target], train.predicted))
print('score', score)

test[target] = result

# print(test.head())
# print(test.describe())

predictions = test[['Id', target]]

predictions.to_csv('submission.csv', index=False)

print('%.0f mins' % ((time() - start_time) / 60))
