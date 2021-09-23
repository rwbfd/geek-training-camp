# coding = 'utf-8'
import io
import multiprocessing
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, asdict
import hyperopt.pyll
from hyperopt import fmin, tpe, hp

import numpy as np

import lightgbm as lgb
import xgboost as xgb
import catboost as cat

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import torch
import copy

cpu_count = multiprocessing.cpu_count()
use_gpu = torch.cuda.is_available()


@dataclass
class XGBOpt:
    nthread: any = hp.choice('nthread', [cpu_count])
    eval_metric: any = hp.choice('eval_metric', ['error'])
    booster: any = hp.choice('booster', ['gbtree', 'dart'])
    sample_type: any = hp.choice('sample_type', ['uniform', 'weighted'])
    rate_drop: any = hp.uniform('rate_drop', 0, 0.2)
    objective: any = hp.choice('objective', ['binary:logistic'])
    max_depth: any = hp.choice('max_depth', [4, 5, 6, 7, 8])
    num_round: any = hp.choice('num_round', [100])
    eta: any = hp.uniform('eta', 0.01, 0.1)
    subsample: any = hp.uniform('subsample', 0.8, 1)
    colsample_bytree: any = hp.uniform('colsample_bytree', 0.3, 1)
    gamma: any = hp.choice('gamma', [0, 1, 5])
    min_child_weight: any = hp.uniform('min_child_weight', 0, 15)
    sampling_method: any = hp.choice('sampling_method', ['uniform', 'gradient_based'])

    @staticmethod
    def get_common_paarams():
        return {'nthread': 4, 'max_depth': 3, 'eval_metric': 'error', 'object': 'binary:logistic', 'eta': 0.01,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'num_round': 1000}


@dataclass
class LGBOpt:
    num_threads: any = hp.choice('num_threads', [cpu_count])
    num_leaves: any = hp.choice('num_leaves', [64])
    metric: any = hp.choice('metric', ['binary_error'])
    num_round: any = hp.choice('num_rounds', [1000])
    objective: any = hp.choice('objective', ['binary'])
    learning_rate: any = hp.uniform('learning_rate', 0.01, 0.1)
    feature_fraction: any = hp.uniform('feature_fraction', 0.5, 1.0)
    bagging_fraction: any = hp.uniform('bagging_fraction', 0.8, 1.0)
    device_type: any = hp.choice('device_tpye', ['gpu']) if use_gpu else hp.choice('device_type',
                                                                                   ['cpu'])
    boosting: any = hp.choice('boosting', ['gbdt', 'dart', 'goss'])
    extra_trees: any = hp.choice('extra_tress', [False, True])
    drop_rate: any = hp.uniform('drop_rate', 0, 0.2)
    uniform_drop: any = hp.choice('uniform_drop', [True, False])
    lambda_l1: any = hp.uniform('lambda_l1', 0, 10)  # TODO: Check range
    lambda_l2: any = hp.uniform('lambda_l2', 0, 10)  # TODO: Check range
    min_gain_to_split: any = hp.uniform('min_gain_to_split', 0, 1)  # TODO: Check range
    min_data_in_bin = hp.choice('min_data_in_bin', [3, 5, 10, 15, 20, 50])

    @staticmethod
    def get_common_params():
        return {'num_thread': 4, 'num_leaves': 12, 'metric': 'binary', 'objective': 'binary',
                'num_round': 1000, 'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8}


@dataclass
class CATOpt:
    thread_count: any = hp.choice('thread_count', [cpu_count])
    num_round: any = hp.choice('num_round', [100])
    objective: any = hp.choice('objective', ['CrossEntropy'])
    eval_metric: any = hp.choice('eval_metric', ['Accuracy'])
    learning_rate: any = hp.uniform('learning_rate', 0.01, 0.1)
    l2_leaf_reg: any = hp.uniform('l2_leaf_reg', 0, 10)  # TODO: Check range
    bootstrap_type: any = hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
    nan_mode: any = hp.choice('nan_mode', ['Forbidden', 'Min', 'Max'])
    leaf_estimation_method: any = hp.choice('leaf_estimation_method', ['Newton', 'Gradient'])
    depth: any = hp.choice('depth', [2, 3, 4, 5, 6, 7, 8])  # TODO: Check range
    max_bin: any = hp.choice('max_bin', [3, 5, 10, 15, 20, 50, 100, 500])

    @staticmethod
    def get_common_params():
        return {'thread_count': 4, 'num_round': 100, 'learning_rate': 0.01, 'objective': 'CrossEntropy',
                'eval_metric': 'Accuracy', 'depth': 3}


@dataclass
class LROpt:
    penalty: any = hp.choice('penalty', ['l2'])
    C: any = hp.uniform('C', 0.5, 10)
    max_iter: any = hp.choice('max_iter', [1000])
    solver: any = hp.choice('solver', ['lbfgs'])

    '''
    penalty: hyperopt.pyll.base.Apply = hp.choice('penalty', ['l2', 'none'])
    solver: hyperopt.pyll.base.Apply = hp.choice('solver', ['lbfgs'])
    C: hyperopt.pyll.base.Apply = hp.uniform('C', 0.5, 2)
    intercept_scaling: hyperopt.pyll.base.Apply = hp.uniform('intercept_scaling', 0.5, 2)
    class_weight: hyperopt.pyll.base.Apply = hp.choice('class_weight', ['balanced'])
    max_iter: hyperopt.pyll.base.Apply = hp.choice('max_iter', [10000])
    n_jobs: hyperopt.pyll.base.Apply = hp.choice('n_jobs', [4])
    '''

    @staticmethod
    def get_common_params():
        return {'penalty': 'l2', 'C': 0.5, 'solver': 'lbfgs'}


@dataclass
class KNNOpt:
    n_neighbors: any = hp.choice('n_neighbors', [2, 5, 10])
    weights: any = hp.choice('weights', ['uniform', 'distance'])
    leaf_size: any = hp.choice('leaf_size', [20, 25, 30, 35, 40, 45, 50])
    p: any = hp.uniform('p', 1, 2)

    @staticmethod
    def get_common_params():
        return {'n_neighbors': 5, 'weights': 'distance'}


@dataclass
class SVMOpt:
    C: any = hp.uniform('C', 0, 10)
    kernel: any = hp.choice('kernel', ['rbf', 'sigmoid'])
    gamma: any = hp.choice('gamma', ['scale', 'auto'])
    probability: any = hp.choice('probability', [True])

    @staticmethod
    def get_common_params():
        return {"C": 0, 'kernel': 'rbf', 'probability': True}

    '''
    n_neighbors: hyperopt.pyll.base.Apply = hp.choice('n_neighbors', [2])
    leaf_size: hyperopt.pyll.base.Apply = hp.choice('leaf_size', [20, 25, 30, 35, 40, 45, 50])
    p: hyperopt.pyll.base.Apply = hp.choice('p', [1, 2, 3, 4, 5, 6])
    n_jobs: hyperopt.pyll.Apply = hp.choice('n_jobs', [4])
    algorithm: hyperopt.pyll.Apply = hp.choice('algorithm', ['ball_tree', 'kd_tree', 'brute'])
    '''


@dataclass
class RFOpt:
    num_threads: any = hp.choice('num_threads', [cpu_count])
    num_leaves: any = hp.choice('num_leaves', [64])
    metric: any = hp.choice('metric', ['binary_error'])
    num_round: any = hp.choice('num_rounds', [1000])
    objective: any = hp.choice('objective', ['binary'])
    learning_rate: any = hp.uniform('learning_rate', 0.01, 0.1)
    feature_fraction: any = hp.uniform('feature_fraction', 0.5, 1.0)
    bagging_fraction: any = hp.uniform('bagging_fraction', 0.8, 1.0)
    device_type: any = hp.choice('device_tpye', ['gpu']) if use_gpu else hp.choice('device_type',
                                                                                   ['cpu'])
    boosting: any = hp.choice('boosting', ['rf'])
    extra_trees: any = hp.choice('extra_tress', [False, True])
    uniform_drop: any = hp.choice('uniform_drop', [True, False])
    lambda_l1: any = hp.uniform('lambda_l1', 0, 10)  # TODO: Check range
    lambda_l2: any = hp.uniform('lambda_l2', 0, 10)  # TODO: Check range
    min_gain_to_split: any = hp.uniform('min_gain_to_split', 0, 1)  # TODO: Check range
    min_data_in_bin = hp.choice('min_data_in_bin', [3, 5, 10, 15, 20, 50])

    @staticmethod
    def get_common_params():
        return {"num_threads": 4, 'num_leaves': 12, 'objective': 'binary', 'metric': 'binary_error',
                'learning_rate': 0.1}


class FitterBase(object):
    def __init__(self, label, metric, max_eval=100, opt=None):
        self.label = label
        self.metric = metric
        self.opt_params = dict()
        self.max_eval = max_eval
        self.opt = opt

    def get_loss(self, y, y_pred):
        if self.metric == 'error':
            return 1 - accuracy_score(y, y_pred)
        elif self.metric == 'precision':
            return 1 - precision_score(y, y_pred)
        elif self.metric == 'recall':
            return 1 - recall_score(y, y_pred)
        elif self.metric == 'macro_f1':
            return 1 - f1_score(y, y_pred, average='macro')
        elif self.metric == 'micro_f1':
            return 1 - f1_score(y, y_pred, average='micro')
        elif self.metric == 'auc':  # TODO: Add a warning checking if y_predict is all [0, 1], it should be probability
            return 1 - roc_auc_score(y, y_pred)
        else:
            raise Exception("Not implemented yet.")

    def get_rand_param(self):
        if self.opt is None:
            return None
        else:
            return hyperopt.pyll.stochastic.sample(asdict(self.opt))


class XGBFitter(FitterBase):
    def __init__(self, label='label', metric='error', opt: XGBOpt = None, max_eval=100):
        super(XGBFitter, self).__init__(label, metric, max_eval)
        if opt is not None:
            self.opt = opt
        else:
            self.opt = XGBOpt()
        self.best_round = None
        self.clf = None

    def train(self, train_df, eval_df, params=None, use_best_eval=True):
        self.best_round = None
        dtrain = xgb.DMatrix(train_df.drop(columns=[self.label]), train_df[self.label])
        deval = xgb.DMatrix(eval_df.drop(columns=[self.label]), eval_df[self.label])
        evallist = [(dtrain, 'train'), (deval, 'eval')]
        if params is None:
            use_params = deepcopy(self.opt_params)
        else:
            use_params = deepcopy(params)

        num_round = use_params.pop('num_round')
        if use_best_eval:
            with io.StringIO() as buf, redirect_stdout(buf):
                self.clf = xgb.train(use_params, dtrain, num_round, evallist)
                output = buf.getvalue().split("\n")
            min_error = np.inf
            min_index = 0
            for idx in range(len(output) - 1):
                if len(output[idx].split("\t")) == 3:
                    temp = float(output[idx].split("\t")[2].split(":")[1])
                    if min_error > temp:
                        min_error = temp
                        min_index = int(output[idx].split("\t")[0][1:-1])
            print("The minimum is attained in round %d" % (min_index + 1))
            self.best_round = min_index + 1
            return output
        else:
            with io.StringIO() as buf, redirect_stdout(buf):
                self.clf = xgb.train(use_params, dtrain, num_round, evallist)
                output = buf.getvalue().split("\n")
                self.best_round = num_round
            return output

    def search(self, train_df, eval_df, use_best_eval=True):
        self.opt_params = dict()

        def train_impl(params):
            self.train(train_df, eval_df, params, use_best_eval)
            if self.metric == 'auc':
                y_pred = self.clf.predict(xgb.DMatrix(eval_df.drop(columns=[self.label])), ntree_limit=self.best_round)
            else:
                y_pred = (self.clf.predict(xgb.DMatrix(eval_df.drop(columns=[self.label])),
                                           ntree_limit=self.best_round) > 0.5).astype(int)
            return self.get_loss(eval_df[self.label], y_pred)

        self.opt_params = fmin(train_impl, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def search_k_fold(self, k_fold, data, use_best_eval=True):
        self.opt_params = dict()

        def train_impl_nfold(params):
            loss = list()
            for train_id, eval_id in k_fold.split(data):
                train_df = data.loc[train_id]
                eval_df = data.loc[eval_id]
                self.train(train_df, eval_df, params, use_best_eval)
                if self.metric == 'auc':
                    y_pred = self.clf.predict(xgb.DMatrix(eval_df.drop(columns=[self.label])),
                                              ntree_limit=self.best_round)
                else:
                    y_pred = (self.clf.predict(xgb.DMatrix(eval_df.drop(columns=[self.label])),
                                               ntree_limit=self.best_round) > 0.5).astype(int)
                loss.append(self.get_loss(eval_df[self.label], y_pred))
            return np.mean(loss)

        self.opt_params = fmin(train_impl_nfold, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def train_k_fold(self, k_fold, train_data, test_data, params=None, drop_test_y=True, use_best_eval=True):
        acc_result = list()
        result_models = list()
        train_pred = np.empty(train_data.shape[0])
        test_pred = np.empty(test_data.shape[0])
        if drop_test_y:
            dtest = test_data.drop(columns=self.label)
        else:
            dtest = test_data
        for train_id, eval_id in k_fold.split(train_data):
            train_df = train_data.loc[train_id]
            eval_df = train_data.loc[eval_id]
            self.train(train_df, eval_df, params, use_best_eval)
            result_models.append(copy.deepcopy(self.clf))
            train_pred[eval_id] = self.clf.predict(xgb.DMatrix(eval_df.drop(columns=self.label)),
                                                   ntree_limit=self.best_round)
            if self.metric == 'auc':
                y_pred = self.clf.predict(xgb.DMatrix(eval_df.drop(columns=[self.label])),
                                          ntree_limit=self.best_round)
            else:
                y_pred = (self.clf.predict(xgb.DMatrix(eval_df.drop(columns=[self.label])),
                                           ntree_limit=self.best_round) > 0.5).astype(int)
            acc_result.append(self.get_loss(eval_df[self.label], y_pred))
            test_pred += self.clf.predict(xgb.DMatrix(dtest), ntree_limit=self.best_round)
        test_pred /= k_fold.n_splits
        return train_pred, test_pred, acc_result, result_models


class LGBFitter(FitterBase):
    def __init__(self, label='label', metric='error', opt: LGBOpt = None, max_eval=100):
        super(LGBFitter, self).__init__(label, metric, max_eval)
        if opt is not None:
            self.opt = opt
        else:
            self.opt = LGBOpt()
        self.best_round = None
        self.clf = None

    def train(self, train_df, eval_df, params=None, use_best_eval=True):
        self.best_round = None
        dtrain = lgb.Dataset(train_df.drop(columns=[self.label]), train_df[self.label])
        deval = lgb.Dataset(eval_df.drop(columns=[self.label]), eval_df[self.label])
        evallist = [dtrain, deval]
        if params is None:
            use_params = deepcopy(self.opt_params)
        else:
            use_params = deepcopy(params)

        num_round = use_params.pop('num_round')
        if use_best_eval:
            with io.StringIO() as buf, redirect_stdout(buf):
                self.clf = lgb.train(use_params, dtrain, num_round, valid_sets=evallist)
                output = buf.getvalue().split("\n")
            min_error = np.inf
            min_index = 0
            for idx in range(len(output) - 1):
                if len(output[idx].split("\t")) == 3:
                    temp = float(output[idx].split("\t")[2].split(":")[1])
                    if min_error > temp:
                        min_error = temp
                        min_index = int(output[idx].split("\t")[0][1:-1])
            print("The minimum is attained in round %d" % (min_index + 1))
            self.best_round = min_index + 1
            return output
        else:
            with io.StringIO() as buf, redirect_stdout(buf):
                self.clf = lgb.train(use_params, dtrain, num_round, valid_sets=evallist)
                output = buf.getvalue().split("\n")
            self.best_round = num_round
            return output

    def search(self, train_df, eval_df, use_best_eval=True):
        self.opt_params = dict()

        def train_impl(params):
            self.train(train_df, eval_df, params, use_best_eval)
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]), num_iteration=self.best_round)
            else:
                y_pred = (self.clf.predict(eval_df.drop(columns=[self.label]),
                                           num_iteration=self.best_round) > 0.5).astype(int)
            return self.get_loss(eval_df[self.label], y_pred)

        self.opt_params = fmin(train_impl, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def search_k_fold(self, k_fold, data, use_best_eval=True):
        self.opt_params = dict()

        def train_impl_nfold(params):
            loss = list()
            for train_id, eval_id in k_fold.split(data):
                train_df = data.loc[train_id]
                eval_df = data.loc[eval_id]
                self.train(train_df, eval_df, params, use_best_eval)
                if self.metric == 'auc':
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label]), num_iteration=self.best_round)
                else:
                    y_pred = (self.clf.predict(eval_df.drop(columns=[self.label]),
                                               num_iteration=self.best_round) > 0.5).astype(int)
                loss.append(self.get_loss(eval_df[self.label], y_pred))
            return np.mean(loss)

        self.opt_params = fmin(train_impl_nfold, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def train_k_fold(self, k_fold, train_data, test_data, params=None, drop_test_y=True, use_best_eval=True):
        acc_result = list()
        train_pred = np.empty(train_data.shape[0])
        test_pred = np.empty(test_data.shape[0])
        if drop_test_y:
            dtest = test_data.drop(columns=self.label)
        else:
            dtest = test_data

        models = list()
        for train_id, eval_id in k_fold.split(train_data):
            train_df = train_data.loc[train_id]
            eval_df = train_data.loc[eval_id]
            self.train(train_df, eval_df, params, use_best_eval)
            models.append(copy.deepcopy(self.clf))
            train_pred[eval_id] = self.clf.predict(eval_df.drop(columns=self.label), num_iteration=self.best_round)
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]), num_iteration=self.best_round)
            else:
                y_pred = (self.clf.predict(eval_df.drop(columns=[self.label]),
                                           num_iteration=self.best_round) > 0.5).astype(int)
            acc_result.append(self.get_loss(eval_df[self.label], y_pred))
            test_pred += self.clf.predict(dtest, num_iteration=self.best_round)
        test_pred /= k_fold.n_splits
        return train_pred, test_pred, acc_result, models


class CATFitter(FitterBase):
    def __init__(self, label='label', metric='error', opt: CATOpt = None, max_eval=100):
        super(CATFitter, self).__init__(label, metric, max_eval)
        if opt is not None:
            self.opt = opt
        else:
            self.opt = CATOpt()
        self.best_round = None
        self.clf = None

    def train(self, train_df, eval_df, params=None, use_best_eval=True):
        self.best_round = None
        dtrain = cat.Pool(data=train_df.drop(columns=[self.label]), label=train_df[self.label])
        deval = cat.Pool(data=eval_df.drop(columns=[self.label]), label=eval_df[self.label])
        if params is None:
            use_params = deepcopy(self.opt_params)
        else:
            use_params = deepcopy(params)
        num_round = use_params.pop('num_round')
        if use_best_eval:
            with io.StringIO() as buf, redirect_stdout(buf):
                self.clf = cat.train(params=use_params,
                                     pool=dtrain,
                                     evals=deval,
                                     num_boost_round=num_round
                                     )

                output = buf.getvalue().split("\n")
            min_error = np.inf
            min_index = 0

            for idx in range(1, num_round + 1):
                if len(output[idx].split("\t")) == 6:
                    temp = 1 - float(output[idx].split("\t")[2].split(":")[1])
                    if min_error > temp:
                        min_error = temp
                        min_index = int(output[idx].split("\t")[0][:-1])
            print("The minimum is attained in round %d" % (min_index + 1))
            self.best_round = min_index + 1
            return output
        else:
            with io.StringIO() as buf, redirect_stdout(buf):
                self.clf = cat.train(params=use_params,
                                     pool=dtrain,
                                     evals=deval,
                                     num_boost_round=num_round
                                     )
                output = buf.getvalue().split("\n")
            self.best_round = num_round
            return output

    def search(self, train_df, eval_df, use_best_eval=True):
        self.opt_params = dict()

        def train_impl(params):
            self.train(train_df, eval_df, params, use_best_eval)
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]),
                                          ntree_end=self.best_round - 1)
            else:
                y_pred = (self.clf.predict(eval_df.drop(columns=[self.label]),
                                           ntree_end=self.best_round - 1) > 0.5).astype(int)
            return self.get_loss(eval_df[self.label], y_pred)

        self.opt_params = fmin(train_impl, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def search_k_fold(self, k_fold, data, use_best_eval=True):
        self.opt_params = dict()

        def train_impl_nfold(params):
            loss = list()
            for train_id, eval_id in k_fold.split(data):
                train_df = data.loc[train_id]
                eval_df = data.loc[eval_id]
                self.train(train_df, eval_df, params, use_best_eval)
                if self.metric == 'auc':
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label]),
                                              ntree_end=self.best_round - 1)
                else:
                    y_pred = (self.clf.predict(eval_df.drop(columns=[self.label]),

                                               ntree_end=self.best_round - 1) > 0.5).astype(int)
                loss.append(self.get_loss(eval_df[self.label], y_pred))
            return np.mean(loss)

        self.opt_params = fmin(train_impl_nfold, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def train_k_fold(self, k_fold, train_data, test_data, params=None, drop_test_y=True, use_best_eval=True):
        acc_result = list()
        train_pred = np.empty(train_data.shape[0])
        test_pred = np.empty(test_data.shape[0])
        if drop_test_y:
            dtest = test_data.drop(columns=self.label)
        else:
            dtest = test_data
        models = list()
        for train_id, eval_id in k_fold.split(train_data):
            train_df = train_data.loc[train_id]
            eval_df = train_data.loc[eval_id]
            self.train(train_df, eval_df, params, use_best_eval)
            models.append(copy.deepcopy(self.clf))
            train_pred[eval_id] = self.clf.predict(eval_df.drop(columns=self.label),
                                                   ntree_end=self.best_round - 1)
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]), ntree_end=self.best_round - 1)
            else:
                y_pred = (self.clf.predict(eval_df.drop(columns=[self.label]),
                                           ntree_end=self.best_round - 1) > 0.5).astype(
                    int)
            acc_result.append(self.get_loss(eval_df[self.label], y_pred))
            test_pred += self.clf.predict(dtest,
                                          ntree_end=self.best_round - 1)
        test_pred /= k_fold.n_splits
        return train_pred, test_pred, acc_result


class LRFitter(FitterBase):
    def __init__(self, label='label', metric='error', opt: LROpt = None, max_eval=100):
        super(LRFitter, self).__init__(label, metric, max_eval)
        if opt is not None:
            self.opt = opt
        else:
            self.opt = LROpt()
        self.clf = None

    def train(self, train_df, eval_df, params=None):
        x_train, y_train, x_eval, y_eval = train_df.drop(columns=[self.label]), train_df[self.label], \
                                           eval_df.drop(columns=[self.label]), eval_df[self.label],

        if params is None:
            use_params = deepcopy(self.opt_params)
        else:
            use_params = deepcopy(params)
        self.clf = LogisticRegression(**use_params)
        self.clf.fit(X=x_train, y=y_train)
        preds = self.clf.predict(X=x_eval)
        output = self.get_loss(y_pred=preds, y=y_eval)
        return output

    def search(self, train_df, eval_df):
        self.opt_params = dict()

        def train_impl(params):
            self.train(train_df, eval_df, params)
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)

            return self.get_loss(eval_df[self.label], y_pred)

        self.opt_params = fmin(train_impl, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def search_k_fold(self, k_fold, data):
        self.opt_params = dict()

        def train_impl_nfold(params):
            loss = list()
            for train_id, eval_id in k_fold.split(data):
                train_df = data.iloc[train_id, :]
                eval_df = data.iloc[eval_id, :]
                self.train(train_df, eval_df, params)
                if self.metric == 'auc':
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
                else:
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)
                loss.append(self.get_loss(eval_df[self.label], y_pred))
            return np.mean(loss)

        self.opt_params = fmin(train_impl_nfold, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def train_k_fold(self, k_fold, train_data, test_data, params=None, drop_test_y=True):
        acc_result = list()
        train_pred = np.empty(train_data.shape[0])
        test_pred = np.empty(test_data.shape[0])
        if drop_test_y:
            dtest = test_data.drop(columns=self.label)
        else:
            dtest = test_data
        for train_id, eval_id in k_fold.split(train_data):
            train_df = train_data.iloc[train_id]
            eval_df = train_data.iloc[eval_id]
            self.train(train_df, eval_df, params)
            models.append(deepcopy(self.clf))
            train_pred[eval_id] = self.clf.predict(eval_df.drop(columns=self.label))
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)

            acc_result.append(self.get_loss(eval_df[self.label], y_pred))
            test_pred += self.clf.predict(dtest)
        test_pred /= k_fold.n_splits
        return train_pred, test_pred, acc_result, models


class KNNFitter(FitterBase):
    def __init__(self, label='label', metric='error', opt: KNNOpt = None, max_eval=100):
        super(KNNFitter, self).__init__(label, metric, max_eval)
        if opt is not None:
            self.opt = opt
        else:
            self.opt = KNNOpt()
        self.clf = None

    def train(self, train_df, eval_df, params=None):
        x_train, y_train, x_eval, y_eval = train_df.drop(columns=[self.label]), train_df[self.label], \
                                           eval_df.drop(columns=[self.label]), eval_df[self.label],

        if params is None:
            use_params = deepcopy(self.opt_params)
        else:
            use_params = deepcopy(params)
        self.clf = KNeighborsClassifier(**use_params)
        self.clf.fit(X=x_train, y=y_train)
        preds = self.clf.predict(X=x_eval)
        output = self.get_loss(y_pred=preds, y=y_eval)

        return output

    def search(self, train_df, eval_df):
        self.opt_params = dict()

        def train_impl(params):
            self.train(train_df, eval_df, params)
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)

            return self.get_loss(eval_df[self.label], y_pred)

        self.opt_params = fmin(train_impl, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def search_k_fold(self, k_fold, data):
        self.opt_params = dict()

        def train_impl_nfold(params):
            loss = list()
            for train_id, eval_id in k_fold.split(data):
                train_df = data.iloc[train_id, :]
                eval_df = data.iloc[eval_id, :]
                self.train(train_df, eval_df, params)
                if self.metric == 'auc':
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
                else:
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)
                loss.append(self.get_loss(eval_df[self.label], y_pred))
            return np.mean(loss)

        self.opt_params = fmin(train_impl_nfold, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def train_k_fold(self, k_fold, train_data, test_data, params=None, drop_test_y=True):
        acc_result = list()
        train_pred = np.empty(train_data.shape[0])
        test_pred = np.empty(test_data.shape[0])
        if drop_test_y:
            dtest = test_data.drop(columns=self.label)
        else:
            dtest = test_data
        models = list()
        for train_id, eval_id in k_fold.split(train_data):
            train_df = train_data.iloc[train_id]
            eval_df = train_data.iloc[eval_id]
            self.train(train_df, eval_df, params)
            models.append(copy.deepcopy(self.clf))
            train_pred[eval_id] = self.clf.predict(eval_df.drop(columns=self.label))
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)

            acc_result.append(self.get_loss(eval_df[self.label], y_pred))
            test_pred += self.clf.predict(dtest)
        test_pred /= k_fold.n_splits
        return train_pred, test_pred, acc_result, models


class SVMFitter(FitterBase):
    def __init__(self, label='label', metric='error', opt: SVMOpt = None, max_eval=100):
        super(SVMFitter, self).__init__(label, metric, max_eval)
        if opt is not None:
            self.opt = opt
        else:
            self.opt = SVMOpt()
        self.clf = None

    def train(self, train_df, eval_df, params=None):
        x_train, y_train, x_eval, y_eval = train_df.drop(columns=[self.label]), train_df[self.label], \
                                           eval_df.drop(columns=[self.label]), eval_df[self.label],
        if params is None:
            use_params = deepcopy(self.opt_params)
        else:
            use_params = deepcopy(params)
        self.clf = SVC(**use_params)
        self.clf.fit(X=x_train, y=y_train)
        preds = self.clf.predict(X=x_eval)
        output = self.get_loss(y_pred=preds, y=y_eval)

        return output

    def search(self, train_df, eval_df):
        self.opt_params = dict()

        def train_impl(params):
            self.train(train_df, eval_df, params)
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)

            return self.get_loss(eval_df[self.label], y_pred)

        self.opt_params = fmin(train_impl, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def search_k_fold(self, k_fold, data):
        self.opt_params = dict()

        def train_impl_nfold(params):
            loss = list()
            for train_id, eval_id in k_fold.split(data):
                train_df = data.loc[train_id]
                eval_df = data.loc[eval_id]
                self.train(train_df, eval_df, params)
                if self.metric == 'auc':
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
                else:
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)
                loss.append(self.get_loss(eval_df[self.label], y_pred))
            return np.mean(loss)

        self.opt_params = fmin(train_impl_nfold, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def train_k_fold(self, k_fold, train_data, test_data, params=None, drop_test_y=True):
        acc_result = list()
        train_pred = np.empty(train_data.shape[0])
        test_pred = np.empty(test_data.shape[0])
        if drop_test_y:
            dtest = test_data.drop(columns=self.label)
        else:
            dtest = test_data
        models = list()
        for train_id, eval_id in k_fold.split(train_data):
            train_df = train_data.iloc[train_id, :]
            eval_df = train_data.iloc[eval_id, :]
            self.train(train_df, eval_df, params)
            train_pred[eval_id] = self.clf.predict_proba(eval_df.drop(columns=self.label))[:, 1]
            train_df = train_data.loc[train_id]
            eval_df = train_data.loc[eval_id]
            self.train(train_df, eval_df, params)
            models.append(deepcopy(self.clf))
            train_pred[eval_id] = self.clf.predict(eval_df.drop(columns=self.label))
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)

            acc_result.append(self.get_loss(eval_df[self.label], y_pred))
            test_pred += self.clf.predict_proba(dtest)[:, 1]
        test_pred /= k_fold.n_splits
        return train_pred, test_pred, acc_result, models


class RFitter(FitterBase):
    def __init__(self, label='label', metric='error', opt: RFOpt = None, max_eval=100):
        super(RFitter, self).__init__(label, metric, max_eval)
        if opt is not None:
            self.opt = opt
        else:
            self.opt = RFOpt()
        self.clf = None

    def train(self, train_df, eval_df, params=None):
        x_train, y_train, x_eval, y_eval = train_df.drop(columns=[self.label]), train_df[self.label], \
                                           eval_df.drop(columns=[self.label]), eval_df[self.label],
        if params is None:
            use_params = deepcopy(self.opt_params)
        else:
            use_params = deepcopy(params)
        self.clf = SVC(**use_params)
        self.clf.fit(X=x_train, y=y_train)
        preds = self.clf.predict(X=x_eval)
        output = self.get_loss(y_pred=preds, y=y_eval)

        return output

    def search(self, train_df, eval_df):
        self.opt_params = dict()

        def train_impl(params):
            self.train(train_df, eval_df, params)
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)

            return self.get_loss(eval_df[self.label], y_pred)

        self.opt_params = fmin(train_impl, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def search_k_fold(self, k_fold, data):
        self.opt_params = dict()

        def train_impl_nfold(params):
            loss = list()
            for train_id, eval_id in k_fold.split(data):
                train_df = data.loc[train_id]
                eval_df = data.loc[eval_id]
                self.train(train_df, eval_df, params)
                if self.metric == 'auc':
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
                else:
                    y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)
                loss.append(self.get_loss(eval_df[self.label], y_pred))
            return np.mean(loss)

        self.opt_params = fmin(train_impl_nfold, asdict(self.opt), algo=tpe.suggest, max_evals=self.max_eval)

    def train_k_fold(self, k_fold, train_data, test_data, params=None, drop_test_y=True):
        acc_result = list()
        train_pred = np.empty(train_data.shape[0])
        test_pred = np.empty(test_data.shape[0])
        if drop_test_y:
            dtest = test_data.drop(columns=self.label)
        else:
            dtest = test_data
        models = list()
        for train_id, eval_id in k_fold.split(train_data):
            train_df = train_data.iloc[train_id, :]
            eval_df = train_data.iloc[eval_id, :]
            self.train(train_df, eval_df, params)
            models.append(deepcopy(self.clf))
            train_pred[eval_id] = self.clf.predict_proba(eval_df.drop(columns=self.label))[:, 1]
            train_df = train_data.loc[train_id]
            eval_df = train_data.loc[eval_id]
            self.train(train_df, eval_df, params)
            train_pred[eval_id] = self.clf.predict(eval_df.drop(columns=self.label))
            if self.metric == 'auc':
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label]))
            else:
                y_pred = self.clf.predict(eval_df.drop(columns=[self.label])).astype(int)

            acc_result.append(self.get_loss(eval_df[self.label], y_pred))
            test_pred += self.clf.predict_proba(dtest)[:, 1]
        test_pred /= k_fold.n_splits
        return train_pred, test_pred, acc_result, models
