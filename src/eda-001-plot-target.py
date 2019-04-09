# plot target histogram

import os
import sys  # noqa
from time import time
from pprint import pprint  # noqa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def plot_target(train):
    train['SalePrice'].hist(bins=40)

    plt.show()


# --------------------- run


def run():

    unique_id = 'Id'
    target = 'SalePrice'

    # load data

    train = pd.read_csv(f'../input/{train_file}.csv{zipext}')
    test = pd.read_csv(f'../input/test.csv{zipext}')

    plot_target(train)

# -------- main


run()

print(f'Finished {((time() - start_time) / 60):.1f} mins\a')
