import sys
import random
import pandas as pd
import os
import shutil

def train_val_test(x_strat, y_start, dst):
    X = os.listdir(x_strat)
    Y = pd.read_csv(y_start, delimiter=r'\s+', dtype={'id': object})

    total = len(X)
    train_num = int(total * 0.8)
    val_num = int(total * 0.9)
    start_point = 0
    desk = list(range(total))
    random.shuffle(desk)

    for dir_name, end_point in zip(('train', 'val', 'test'), (train_num, val_num, total)):
        end_path = os.path.join(dst, dir_name)
        os.makedirs(end_path, exist_ok=True)
        for idx in desk[start_point:end_point]:
            # x data prepare
            file_name = X[idx]
            begin = os.path.join(x_strat, file_name)
            end = os.path.join(end_path, file_name)
            shutil.copy(begin, end)
        # y data prepare
        y = Y.iloc[desk[start_point:end_point]]
        y = y.sort_index()
        y.to_csv(os.path.join(dst, '{}.txt'.format(dir_name)), index=False, sep=' ')
        start_point = end_point

if __name__ == '__main__':
    x_strat = sys.argv[1]
    y_start = sys.argv[2]
    dst = sys.argv[3]
    random.seed(42)
    train_val_test(x_strat, y_start, dst)
