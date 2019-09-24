# train/test split
# local score  0.172
# kaggle score 0.196
# minimize score

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

my_model = XGBRegressor(random_state=0)

# Fit the model to the training data & validate for score
my_model.fit(X_train, y_train)
preds_valid = my_model.predict(X_valid)
score = np.sqrt(mean_squared_error(np.log(y_valid), np.log(preds_valid)))

print(f'score: {score}')

# Generate test predictions on full set
my_model.fit(X, y)

preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)
