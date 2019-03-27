import os
import cv2
import random
from collections import defaultdict
import numpy as np
from keras.applications.nasnet import preprocess_input
from keras.optimizers import Adagrad
from keras.utils.vis_utils import plot_model
from lifelines.utils import concordance_index
from models import model_nas, negative_log_likelihood
from tools import Train_table_creator, gen_logger, get_seq
random.seed(42)

class SNAS:
    def __init__(self, selected_p, dst, train_size_ratio=0.8, epochs=40, inner_train_time=22,
    val_sel_num=10, aug_time=10, logger=None, d_size=256):
        self.model = model_nas(d_size=d_size)
        self.dst = dst
        self.logger = logger if logger is not None else gen_logger('SNAS')
        trainer = Train_table_creator(selected_p, dst, train_ratio=train_size_ratio, logger=self.logger)
        assert trainer.cache() == True, 'imgs searching entounters error'
        self.train_table = trainer.train_table
        self.test_table = trainer.test_table
        self.sel_num = val_sel_num
        self.epochs = epochs
        self.start_epoch = 0
        self.inner_train_time = inner_train_time
        self.aug_time = aug_time
        self.trained = False
        self.ada = None 
        self.seq = None
        self.pool = defaultdict(list)
        self.train_pool = defaultdict(list)

    def _get_pool(self, dir_p, bound=0):
        pool = self.pool if bound == 0 else self.train_pool
        if dir_p not in pool:
            pool[dir_p] = os.listdir(dir_p)
        elif len(pool[dir_p]) <= bound:
            pool[dir_p] = os.listdir(dir_p)
        return pool[dir_p]

    def _read_train_dir(self, dir_p):
        # here I try to make sure no repetitive selections in training
        pool = self._get_pool(dir_p)
        sel = random.choice(pool)
        self.pool[dir_p].remove(sel)
        x = os.path.join(dir_p, sel)
        return cv2.imread(x)

    def _read_val_dir(self, dir_p, use_filter=False) -> list:
        pool = self._get_pool(dir_p, bound=self.sel_num)
        pool_size = len(pool)
        if use_filter and pool_size < self.sel_num:
            raise ValueError(f'the number ({pool_size}) in dir is not enough to have a reliable validation')
        sels = random.choices(pool, k=self.sel_num)
        xs = [os.path.join(dir_p, sel) for sel in sels]
        return [cv2.imread(x) for x in xs]

    def _chunk(self, df_sort, batch_size=64):
        population = list(range(len(df_sort)))
        for _ in range(self.epochs):
            chunk_idx = random.choices(population, k=batch_size)
            chunk_idx.sort()
            yield df_sort.iloc[chunk_idx]

    def _data_val(self, df_sort, use_filter=False):
        X, T, E = [], [], []
        for item in df_sort.iterrows():
            path = item[1][0]
            dur = item[1][2]
            obs = item[1][3]
            try:
                X.append(self._read_val_dir(path, use_filter=use_filter))
            except ValueError:
                self.logger.warning(f'check imgs in {path}')
                continue
            else:
                T.append(dur)
                E.append(obs)
        T = np.array(T)
        return X, T, E

    def _data_gen_batch(self, df_sort, batch_size=64):
        for chunk_df in self._chunk(df_sort, batch_size=batch_size):
            X, T, E = [], [], []
            for item in chunk_df.iterrows():
                path = item[1][0]
                dur = item[1][2]
                obs = item[1][3]
                X.append(self._read_train_dir(path))
                T.append(dur)
                E.append(obs)
            T = np.array(T)
            yield X, T, E

    def _data_gen_whole(self, df_sort):
        # whole will not change T and E, so we can save resource
        X, T, E, paths = [], [], [], []
        for item in df_sort.iterrows():
            path = item[1][0]
            dur = item[1][2]
            obs = item[1][3]
            paths.append(path)
            X.append(self._read_train_dir(path))
            T.append(dur)
            E.append(obs)
        T = np.array(T)
        yield X, T, E

        for _ in range(self.epochs-1):
            X.clear()
            for path in paths:
                X.append(self._read_train_dir(path))
            yield X, T, E

    def _x_aug(self, X):
        if self.aug_time == 0:
            yield np.array(X)
        else: 
            for _ in range(self.aug_time):
                X = self.seq.augment_images(X)
                X = [preprocess_input(x) for x in X]
                yield np.array(X)

    def _model_eval(self, X_val, y, e):
        hr_preds = []
        for x_case in X_val:
            x_case = [preprocess_input(x) for x in x_case]
            x_case = np.array(x_case)
            hr_pred = self.model.predict(x_case)
            hr_pred = sorted(hr_pred)[-2] # only the second most serious area
            hr_preds.append(hr_pred)
        hr_preds = np.exp(hr_preds)
        ci = concordance_index(y,-hr_preds,e)
        return ci

    def _train_aux(self, X, Y, event_size=None):
        for X in self._x_aug(X):
            self.model.fit(
                X, Y,
                batch_size=event_size,
                epochs=self.inner_train_time,
                verbose=False,
                shuffle=False)
        self.trained = True

    def _train_init(self):
        if self.ada is None or self.seq is None:
            self.ada = Adagrad(lr=1e-3, decay=0.1)
            self.seq = get_seq()

    def plot(self):
        plot_model(self.model, to_file=f'{self.dst}/model.png') 

    def set_start_epoch(self, start_epoch):
        self.start_epoch = start_epoch

    def whole_train(self):
        # whole train can allocate more resources to train on varying imgs, so epochs can be more
        start = True
        event_size = None
        self._train_init()
        for epoch, (X, Y, E) in enumerate(self._data_gen_whole(self.train_table), start=self.start_epoch):
            if start:
                self.model.compile(loss=negative_log_likelihood(E), optimizer=self.ada)
                event_size = len(E)
                start = False
            self._train_aux(X, Y, event_size=event_size)
            self.model.save_weights(os.path.join(self.dst, f'{epoch}.h5'))
            self.logger.info(f'{epoch} done')
            self.feedback()

    def batch_train(self, batch_size=64):
        event_size = batch_size
        self._train_init()
        for epoch, (X, Y, E) in enumerate(self._data_gen_batch(self.train_table, batch_size=batch_size),start=self.start_epoch):
            self.model.compile(loss=negative_log_likelihood(E), optimizer=self.ada)
            self._train_aux(X, Y, event_size=event_size)
            self.model.save_weights(os.path.join(self.dst, f'{epoch}.h5'))
            self.logger.info(f'{epoch} done')
            self.feedback()

    def load(self, weight_p):
        try:
            self.model.load_weights(weight_p)
        except:
            self.logger.exception('Wrong weight saving file')
        else:
            start_epoch = int(os.path.basename(weight_p).split('.')[0]) + 1
            self.set_start_epoch(start_epoch)
            self.trained = True
            self.logger.info('Loading Successful')

    def feedback(self, sel_num=None):
        if self.trained == False:
            print('Load weight or train model first!')
            return
        if sel_num is not None:
            self.sel_num = sel_num
        X, Y, E = self._data_val(self.train_table, use_filter=True)
        X_val, Y_val, E_val = self._data_val(self.test_table, use_filter=True)
        self.logger.info(f'train:{self._model_eval(X, Y, E)} num:{len(Y)}; val:{self._model_eval(X_val, Y_val, E_val)} num:{len(Y_val)}')