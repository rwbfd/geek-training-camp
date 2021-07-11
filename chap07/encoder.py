# coding = 'utf-8'
import pandas as pd
import numpy as np
import category_encoders as ce
from ..general.util import remove_continuous_discrete_prefix, split_df
import copy

import multiprocessing

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import lightgbm as lgb

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from thermoencoder import ThermoEncoder

cpu_count = multiprocessing.cpu_count()

from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, SpectralClustering, AffinityPropagation, \
    AgglomerativeClustering, DBSCAN, OPTICS, Birch

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.mixture import GaussianMixture
import warnings
import logging


class EncoderBase(object):
    def __init__(self):
        self.trans_ls = list()

    def reset(self):
        self.trans_ls = list()

    def check_var(self, df):
        for _, _, target, _ in self.trans_ls:
            if target not in df.columns:
                raise Exception("The columns to be transformed are not in the dataframe.")


class LabelEncoder(EncoderBase):
    def __init__(self):
        super(LabelEncoder, self).__init__()

    def fit(self, df, targets):
        self.reset()
        for target in targets:
            unique = df[target].unique()
            index = range(len(unique))
            mapping = dict(zip(unique, index))
            self.trans_ls.append((target, mapping))

    def transform(self, df):
        df_copy = df.copy(deep=True)
        for name, mapping in self.trans_ls:
            df_copy[name] = df_copy[name].map(lambda x: mapping[x])
        return df_copy


class NANEncoder(EncoderBase):
    def __init__(self):
        warnings.warn(
            "This is a simple application in order to perform the simpliest imputation. It is strongly suggest to use R's mice package instead. ")
        super().__init__()

    def fit(self, df, targets, method='simple_impute'):
        self.reset()

        for target in targets:
            if method == 'simple_impute':
                if target.startswith("continuous_"):
                    self.trans_ls.append((target, df[target].median()))
                elif target.startswith('discrete_'):
                    self.trans_ls.append((target, df[target].mode()))

    def transform(self, df):
        df_copy = df.copy(deep=True)
        for target, value in self.trans_ls:
            df_copy.loc[pd.isnull(df_copy[target]), target] = df_copy.loc[pd.isnull(df_copy[target]), target].map(
                lambda x: value)
        return df_copy


class ScaleEncoder(EncoderBase):
    def __init__(self):
        super().__init__()

    def fit(self, df, targets, configs):
        """
        :param df: the dataframe to transform
        :param targets: a list of variables to perform the scaling
        :param configs: the scaling methods
        """
        for target in targets:
            for method, param in configs:
                if method == 'std':
                    self._fit_standardize(df, method, target, param)
                elif method == 'minmax':
                    self._fit_min_max(df, method, target, param)
                elif method == 'trunc_upper':
                    self._fit_trunc_upper_quantile(df, method, target, param)
                elif method == 'trunc_lower':
                    self._fit_trunc_lower_quantile(df, method, target, param)
                elif method == 'trunc_lower_upper':
                    self._fit_trunc_lowerupper_quantile(df, method, target, param)
                else:
                    raise NotImplementedError()

    def _fit_standardize(self, method, df, target, param):
        mean = df[target].mean()
        std = df[target].std()
        param_dict = {'mean': mean, 'std': std}
        name = target + '_std'
        self.trans_ls.append((method, target, name, param_dict))

    def _fit_min_max(self, df, method, target, param):
        min_value = df[target].min()
        max_value = df[target].max()

        param_dict = {'min': min_value, 'max': max_value}

        name = target + '_minmax'
        self.trans_ls.append((method, target, name, param_dict))

    def _fit_trunc_upper_quantile(self, df, method, target, param):
        upper_quantile = df[target].quantile(param['upper_quantile'])
        name = target + '_upper_q'
        self.trans_ls.append((method, target, name, upper_quantile))

    def _fit_trunc_lower_quantile(self, df, method, target, param):
        lower_quantile = df[target].quantile(param['lower_quantile'])
        name = target + '_lower_q'
        self.trans_ls.append((method, target, name, lower_quantile))

    def _fit_trunc_lowerupper_quantile(self, df, method, target, param):
        lower_quantile = df[target].quantile(param['lower_quantile'])
        upper_quantile = df[target].quantile(param['upper_quantile'])

        param_dict = {'lower_quantile': lower_quantile, upper_quantile: 'upper_quantile'}
        name = target + "_lower_upper_q"
        self.trans_ls.append((method, target, name, param_dict))

    def transform(self, df):
        """
        Perform the transformation based on the previous results.
        """

        df_copy = df.copy(deep=True)
        for method, target, name, param in self.trans_ls:
            if method == 'std':
                df_copy[name] = (df_copy[target] - param['mean']) / param['std']
            elif method == 'minmax':
                df_copy[name] = (df_copy[target] + param['min']) / (param['max'] - param['min'])
            elif method == 'trunc_upper':
                df_copy[name] = df_copy[target].apply(lambda x: x if x <= param else param)
            elif method == 'trunc_lower':
                df_copy[name] = df_copy[target].apply(lambda x: x if x >= param else param)
            elif method == 'trunc_lower_upper':
                def trunc(x):
                    if x >= param['upper_quantile']:
                        return param['upper_quantile']
                    elif x <= param['lower_quantile']:
                        return param['lower_quantile']
                    else:
                        return x

                df_copy[name] = df_copy[target].apply(trunc)
            else:
                raise NotImplementedError()
        return df_copy


class ClusteringEncoder(EncoderBase):
    """
    """

    def __init__(self):
        super().__init__()

    def fit(self, df, targets, configs):
        """
        :param df: the dataframe to train the clustering algorithm.
        :param targets: a list of list of variables.
        :param config: configurations for clustering algorithms
        DBSCAN, OPTICS, Birch
        """
        self.reset()
        for target in targets:
            for config in configs:
                method = config['method']

                if method == 'kmeans':
                    self._fit_kmeans(df, target, config)
                elif method == 'meanshift':
                    self._fit_meanshit(df, target, config)
                elif method == 'affinitypropagation':
                    self._fit_affinitypropagation(df, target, config)
                elif method == 'spectralclustering':
                    self._fit_spectralclustering(df, target, config)
                elif method == 'agglomerativeclustering':
                    self._fit_agglomerativeclustering(df, target, config)
                elif method == 'DBSCAN':
                    self._fit_DBSCAN(df, target, config)
                elif method == 'OPTICS':
                    self._fit_OPTICS(df, target, config)
                elif method == 'birch':
                    self._fit_birch(df, target, config)
                elif method == 'gaussianmixture':
                    self._fit_gaussianmixture(df, target, config)
                elif method == 'latentdirichletallocation':
                    self._fit_latentdirichletallocation(df, target, config)
                else:
                    raise NotImplementedError()

    def _fit_kmeans(self, df, target, config):
        config_cp = copy.deepcopy(config)
        del config_cp['method']
        kmeans = KMeans(**config_cp).fit(df[target])
        name = "_".join(target) + "_kmeans"
        self.trans_ls.append(('kmeans', name, target, kmeans))

    def _fit_meanshit(self, df, target, config):
        config_cp = copy.deepcopy(config)
        bandwidth = estimate_bandwidth(df, config_cp['quantile'], config_cp['n_samples'])
        del config['quantile']
        del config['n_samples']
        del config_cp['method']

        config_cp['bandwidth'] = bandwidth
        encoder = MeanShift(**config_cp).fit(df[target])
        name = "_".join(target) + "_meanshift"
        self.trans_ls.append(('meanshift', name, target, encoder))

    def _fit_affinitypropagation(self, df, target, config):
        config_cp = copy.deepcopy(config)
        del config_cp['method']
        encoder = AffinityPropagation(**config_cp).fit(df[target])
        name = "_".join(target) + "_affinitypropagation"
        self.trans_ls.append(('affinitypropagation', name, target, encoder))

    def _fit_spectralclustering(self, df, target, config):
        config_cp = copy.deepcopy(config)
        del config_cp['method']
        encoder = SpectralClustering(**config_cp).fit(df[target])
        name = "_".join(target) + "_spectralclustering"
        self.trans_ls.append(('spectralclustering', name, target, encoder))

    def _fit_agglomerativeclustering(self, df, target, config):
        config_cp = copy.deepcopy(config)
        del config_cp['method']
        encoder = AgglomerativeClustering(**config_cp).fit(df[target])
        name = "_".join(target) + "_agglomerativeclustering"
        self.trans_ls.append(('agglomerativeclustering', name, target, encoder))

    def _fit_DBSCAN(self, df, target, config):
        config_cp = copy.deepcopy(config)
        del config_cp['method']
        encoder = DBSCAN(**config_cp).fit(df[target])
        name = "_".join(target) + "_DBSCAN"
        self.trans_ls.append(('DBSCAN', name, target, encoder))

    def _fit_OPTICS(self, df, target, config):
        config_cp = copy.deepcopy(config)
        del config_cp['method']
        encoder = OPTICS(**config_cp).fit(df[target])
        name = "_".join(target) + "_OPTICS"
        self.trans_ls.append(('OPTICS', name, target, encoder))

    def _fit_birch(self, df, target, config):
        config_cp = copy.deepcopy(config)
        del config_cp['method']
        encoder = Birch(**config_cp).fit(df[target])
        name = "_".join(target) + "_birch"
        self.trans_ls.append(('birch', name, target, encoder))

    def _fit_gaussianmixture(self, df, target, config):
        config_cp = copy.deepcopy(config)
        del config_cp['method']
        encoder = GaussianMixture(**config_cp).fit(df[target])
        name = "_".join(target) + "_gaussianmixture"
        self.trans_ls.append(('gaussianmixture', name, target, encoder))

    def _fit_latentdirichletallocation(self, df, target, config):
        config_cp = copy.deepcopy(config)
        del config_cp['method']
        encoder = LatentDirichletAllocation(**config_cp).fit(df[target])
        name = "_".join(target) + "_latentdirichletallocation"
        self.trans_ls.append(('latentdirichletallocation', name, target, encoder))

    def transform(self, df):
        df_copy = df.copy
        for method, name, target, encoder in self.trans_ls:
            if method in ['kmeans', 'meanshift', 'affinitypropagation', 'spectralclustering',
                          'agglomerativeclustering', 'DBSCAN', 'OPTICS', 'birch', 'gaussianmixture',
                          'latentdirichletallocation']:
                df_copy[name] = encoder.predict(df_copy[target])
            else:
                raise NotImplementedError()


class CategoryEncoder(EncoderBase):
    def __init__(self):
        super().__init__()

    def fit(self, df, y, targets, configurations):
        """

        :param df: the data frame to be fitted; can be different from the transformed ones.
        :param y: the y variable
        :param targets: the variables to be transformed
        :param configurations: in the form of a list of (method, parameter), where method is one of ['woe', 'one-hot','ordinal','hash'],
        and parameter is a dictionary pertained to each encoding method
        :return:
        """
        self.reset()
        for target in targets:
            for config in configurations:
                self._fit_one(df, y, target, config)

    def _fit_one(self, df, y, target, config):
        method, parameter = config[0], config[1]
        if method == 'woe':
            self._fit_woe(df, y, target)
        elif method == 'one-hot':
            self._fit_one_hot(df, target)
        elif method == 'ordinal':
            self._fit_ordinal(df, target)
        elif method == 'hash':
            self._fit_hash(df, target)
        elif method == 'target':
            self._fit_target(df, y, target, parameter)
        elif method == 'catboost':
            self._fit_catboost(df, y, target, parameter)
        elif method == 'glm':
            self._fit_glm(df, y, target, parameter)
        elif method == 'js':
            self._fit_js(df, y, target, parameter)
        elif method == 'leave_one_out':
            self._fit_leave_one_out(df, y, target, parameter)
        elif method == 'polinomial':
            self._fit_polynomial(df, y, target, parameter)
        elif method == 'sum':
            self._fit_sum(df, y, target, parameter)
        elif method == 'thermo':
            self._fit_thermo(df, y, target, parameter)
        else:
            logging.error("The method you input is %s, and is not supported." % method)
            raise NotImplementedError()

    def _fit_polynomial(self, df, y, target, parameter):
        poly_encoder = ce.PolynomialEncoder()

        poly_encoder.fit(df[target].map(to_str), df[y])
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_poly' for x in
                poly_encoder.get_feature_names()]
        self.trans_ls.append(('polynomial', name, target, poly_encoder))

    def _fit_sum(self, df, y, target, parameter):
        sum_encoder = ce.SumEncoder()

        sum_encoder.fit(df[target].map(to_str))
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_sum' for x in
                sum_encoder.get_feature_names()]
        self.trans_ls.append(('sum_encoder', name, target, sum_encoder))

    def _fit_js(self, df, y, target, parameter):
        js_encoder = ce.JamesSteinEncoder()

        js_encoder.fit(df[target].map(to_str), df[y])
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_js' for x in
                js_encoder.get_feature_names()]
        self.trans_ls.append(('jsencoder', name, target, js_encoder))

    def _fit_leave_one_out(self, df, y, target, parameter):
        loo_encoder = ce.LeaveOneOutEncoder()

        loo_encoder.fit(df[target].map(to_str), df[y])
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_leave_one_out' for x in
                loo_encoder.get_feature_names()]
        self.trans_ls.append(('leave_one_out', name, target, loo_encoder))

    def _fit_catboost(self, df, y, target, parameter):
        cat_encoder = ce.CatBoostEncoder()

        cat_encoder.fit(df[target].map(to_str), df[y])
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_catboost' for x in
                cat_encoder.get_feature_names()]
        self.trans_ls.append(('catboost', name, target, cat_encoder))

    def _fit_glm(self, df, y, target, parameter):
        glm_encoder = ce.GLMMEncoder()

        glm_encoder.fit(df[target].map(to_str), df[y])
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_glm' for x in
                glm_encoder.get_feature_names()]
        self.trans_ls.append(('glm', name, target, glm_encoder))

    def _fit_hash(self, df, target):
        hash_encoder = ce.HashingEncoder()
        hash_encoder.fit(df[target].map(to_str))
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_hash' for x in
                hash_encoder.get_feature_names()]
        self.trans_ls.append(('hash', name, target, hash_encoder))

    def _fit_ordinal(self, df, target):
        ordinal_encoder = ce.OrdinalEncoder()
        ordinal_encoder.fit(df[target].map(to_str))
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_ordinal' for x in
                ordinal_encoder.get_feature_names()]
        self.trans_ls.append(('ordinal', name, target, ordinal_encoder))

    def _fit_target(self, df, y, target, parameter):
        smoothing = parameter['smoothing']
        target_encoder = ce.TargetEncoder(smoothing=smoothing)
        target_encoder.fit(df[target].map(to_str), df[y])
        name = ['continuous_' + remove_continuous_discrete_prefix(x) + '_smooth_' + str(smoothing) + '_target' for x in
                target_encoder.get_feature_names()]
        self.trans_ls.append(('target', name, target, target_encoder))

    def _fit_one_hot(self, df, target):
        one_hot_encoder = ce.OneHotEncoder()
        target_copy = df[target].copy(deep=True)
        target_copy = target_copy.map(to_str)
        one_hot_encoder.fit(target_copy)
        name = [x + "_one_hot" for x in
                one_hot_encoder.get_feature_names()]  ## I assume that the variables start with discrete
        self.trans_ls.append(('one-hot', name, target, one_hot_encoder))

    def _fit_woe(self, df, y, target):  ##
        woe_encoder = ce.woe.WOEEncoder(cols=target)
        woe_encoder.fit(df[target].map(to_str), df[y].map(to_str))
        name = 'continuous_' + remove_continuous_discrete_prefix(target) + "_woe"
        self.trans_ls.append(('woe', name, target, woe_encoder))

    def transform(self, df, y=None):
        """

        :param df: The data frame to be transformed.
        :param y: The name for y variable. Only used for leave-one-out transform for WOE and Target encoder.
        :return: The transformed dataset
        """
        for _, _, target, _ in self.trans_ls:
            if target not in df.columns:
                raise Exception("The columns to be transformed are not in the dataframe.")
        result_df = df.copy(deep=True)
        for method, name, target, encoder in self.trans_ls:
            if method == 'woe':
                if y is not None:
                    result_df[name] = encoder.transform(df[target].map(to_str), df[y])
                else:
                    result_df[name] = encoder.transform(df[target].map(to_str))
            if method == 'one-hot':
                result_df[name] = encoder.transform(df[target].map(to_str))
            if method == 'target':
                if y:
                    result_df[name] = encoder.transform(df[target].map(to_str), df[y])
                else:
                    result_df[name] = encoder.transform(df[target].map(to_str))
            if method == 'hash':
                result_df[name] = encoder.transform(df[target].map(to_str))
            if method == 'ordinal':
                result_df[name] = encoder.transform(df[target].map(to_str))
            if method in ['catboost', 'glm', 'js', 'leave-one-out', 'sum', 'thermo']:
                result_df[name] = encoder.transform(df[target].map(to_str), df[y])
        return result_df

    def _fit_thermo(self, df, y, target, parameter):
        encoder = ThermoEncoder()

        encoder.fit(df[target].map(to_str))
        name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_thermo'
        self.trans_ls.append(('thermo', name, target, encoder))


class DiscreteEncoder(EncoderBase):
    def __init__(self):
        super(DiscreteEncoder, self).__init__()

    def fit(self, df, targets, configurations):
        """

        :param df: the dataframe to be fitted; can be different from the transformed one;
        :param targets: the variables to be transformed
        :param configurations: in the form of a list of (method, parameter), where method is one of ['quantile', 'uniform'],
            the parameter should contain a key called 'nbins'
        and parameter is a dictionary pertained to each encoding method
        :return:
        """
        self.reset()
        for target in targets:
            for method, parameter in configurations:
                nbins = parameter['nbins']
                self._fit_one(df, target, method, nbins)

    def _fit_one(self, df, target, method, nbins):
        if method == 'uniform':
            intervals = self._get_uniform_intervals(df, target, nbins)
            name = 'discrete_' + remove_continuous_discrete_prefix(target) + "_nbins_" + str(
                nbins) + "_uniform_dis_encoder"

            self.trans_ls.append((target, name, intervals))
        elif method == 'quantile':
            intervals = self._get_quantile_intervals(df, target, nbins)
            name = 'discrete_' + remove_continuous_discrete_prefix(target) + "_nbins_" + str(
                nbins) + "_quantile_dis_encoder"
            self.trans_ls.append((target, name, intervals))
        else:
            raise Exception("Not Implemented Yet")

    def transform(self, df):
        result = df.copy(deep=True)
        for target, _, _ in self.trans_ls:
            if target not in df.columns:
                raise Exception("The columns to be transformed are not in the dataframe.")

        for target, name, intervals in self.trans_ls:
            result[name] = encode_label(result[target].map(lambda x: get_interval(x, intervals)))
        return result

    def _get_uniform_intervals(self, df, target, nbins):
        target_var = df[target]
        minimum = target_var[target_var != -np.inf].min()
        maximum = target_var[target_var != np.inf].max()

        intervals = get_uniform_interval(minimum, maximum, nbins)
        return intervals

    def _get_quantile_intervals(self, df, target, nbins):
        return get_quantile_interval(df[target], nbins)


class UnaryContinuousVarEncoder(EncoderBase):
    def __init__(self):
        super(UnaryContinuousVarEncoder, self).__init__()

    def fit(self, targets, config):
        self.reset()
        for target in targets:
            for method, parameter in config:
                if method == 'power':
                    self._fit_power(target, parameter)
                if method == 'sin':
                    self._fit_sin(target)
                if method == 'cos':
                    self._fit_cos(target)
                if method == 'tan':
                    self._fit_tan(target)
                if method == 'log':
                    self._fit_log(target)
                if method == 'exp':
                    self._fit_exp(target)
                if method == 'abs':
                    self._fit_abs(target)
                if method == 'neg':
                    self._fit_neg(target)
                if method == 'inv':
                    self._fit_inv(target)
                if method == 'sqrt':
                    self._fit_sqrt(target)
                if method == 'box_cox':
                    self._fit_box_cox(target, parameter)
                if method == 'yeo_johnson':
                    self._fit_yeo_johnson(target, parameter)

    def _fit_box_cox(self, target, parameter):
        name = target + str(parameter['lambda']) + '_box_cox'

        def encoder(x):
            lambda_1 = parameter['lambda']
            if x <= 0:
                warnings.warn('Box Cox transformation only applies to positive numbers! Returns 0!')
                return 0
            if lambda_1 != 0:
                return (x ** lambda_1 - 1) / lambda_1
            else:
                return np.log(x)

        self.trans_ls.append(('box_cox', name, target, encoder))

    def _fit_yeo_johnson(self, target, parameter):
        lambda_1 = (parameter['lambda'])
        name = target + str(lambda_1) + '_yeo_johnson'

        def encoder(x):
            if lambda_1 != 0 and x >= 0:
                return ((x + 1) ** lambda_1 - 1) / lambda_1
            elif lambda_1 == 0 and x >= 0:
                return np.log(x + 1)
            elif lambda_1 != 2 and x <= 0:
                return -((-x + 1) ** (2 - lambda_1) - 1) / (2 - lambda_1)
            elif lambda_1 == 2 and x <= 0:
                return -np.log(-x + 1)

        self.trans_ls.append(('yeo_johnson', name, target, encoder))

    def transform(self, df):
        result = df.copy(deep=True)
        for method, new_name, target, encoder in self.trans_ls:
            result[new_name] = result[target].apply(encoder)
        return result

    def _fit_power(self, target, parameter):
        order = parameter['order']
        _power = lambda x: np.power(x, order)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + "_power_" + str(order)
        self.trans_ls.append(('power', new_name, target, _power))

    def _fit_sin(self, target):
        _sin = lambda x: np.sin(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_sin'
        self.trans_ls.append(('sin', new_name, target, _sin))

    def _fit_cos(self, target):
        _cos = lambda x: np.cos(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_cos'
        self.trans_ls.append(('cos', new_name, target, _cos))

    def _fit_tan(self, target):
        _tan = lambda x: np.tan(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_tan'
        self.trans_ls.append(('tan', new_name, target, _tan))

    def _fit_log(self, target):
        _log = lambda x: np.log(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_log'
        self.trans_ls.append(('log', new_name, target, _log))

    def _fit_exp(self, target):
        _exp = lambda x: np.exp(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_exp'
        self.trans_ls.append(('exp', new_name, target, _exp))

    def _fit_abs(self, target):
        _abs = lambda x: np.abs(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_abs'
        self.trans_ls.append(('abs', new_name, target, _abs))

    def _fit_neg(self, target):
        _neg = lambda x: -(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_neg'
        self.trans_ls.append(('neg', new_name, target, _neg))

    def _fit_inv(self, target):
        _inv = lambda x: np.divide(1, x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_inv'
        self.trans_ls.append(('inv', new_name, target, _inv))

    def _fit_sqrt(self, target):
        _sqrt = lambda x: np.sqrt(x)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_sqrt'
        self.trans_ls.append(('sqrt', new_name, target, _sqrt))


class BinaryContinuousVarEncoder(EncoderBase):
    def __init__(self):
        super(BinaryContinuousVarEncoder, self).__init__()

    def fit(self, targets_pairs, config):
        for target1, target2 in targets_pairs:
            for method in config:
                if method == 'add':
                    self._fit_add(target1, target2)
                if method == 'sub':
                    self._fit_sub(target1, target2)
                if method == 'mul':
                    self._fit_mul(target1, target2)
                if method == 'div':
                    self._fit_div(target1, target2)

    def transform(self, df):
        result = df.copy(deep=True)
        for method, new_name, target1, target2, encoder in self.trans_ls:
            result[new_name] = result.apply(lambda row: encoder(row[target1], row[target2]), axis=1)
        return result

    def _fit_add(self, target1, target2):
        _add = lambda x, y: np.add(x, y)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target1) + '_' + remove_continuous_discrete_prefix(
            target2) + '_add'
        self.trans_ls.append(('add', new_name, target1, target2, _add))

    def _fit_sub(self, target1, target2):
        _sub = lambda x, y: x - y
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target1) + '_' + remove_continuous_discrete_prefix(
            target2) + '_sub'
        self.trans_ls.append(('sub', new_name, target1, target2, _sub))

    def _fit_mul(self, target1, target2):
        _mul = lambda x, y: np.multiply(x, y)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target1) + '_' + remove_continuous_discrete_prefix(
            target2) + '_mul'
        self.trans_ls.append(('mul', new_name, target1, target2, _mul))

    def _fit_div(self, target1, target2):
        _div = lambda x, y: np.divide(x, y)
        new_name = 'continuous_' + remove_continuous_discrete_prefix(target1) + '_' + remove_continuous_discrete_prefix(
            target2) + '_div'
        self.trans_ls.append(('div', new_name, target1, target2, _div))


class BoostTreeEncoder(EncoderBase):
    def __init__(self, nthread=None):
        super(BoostTreeEncoder, self).__init__()
        if nthread:
            self.nthread = cpu_count
        else:
            self.nthread = nthread

    def fit(self, df, y, targets_list, config):
        self.reset()
        for method, parameter in config:
            if method == 'xgboost':
                self._fit_xgboost(df, y, targets_list, parameter)
            if method == 'lightgbm':
                self._fit_lightgbm(df, y, targets_list, parameter)

    def _fit_xgboost(self, df, y, targets_list, parameter):
        for targets in targets_list:
            parameter_copy = copy.deepcopy(parameter)
            if 'nthread' not in parameter.keys():
                parameter_copy['nthread'] = self.nthread
            if 'objective' not in parameter.keys():
                if len(np.unique(df[y])) == 2:
                    parameter_copy['objective'] = "binary:logistic"
                else:
                    parameter_copy['objective'] = "multi:softmax"

            num_rounds = parameter['num_boost_round']
            pos = parameter['pos']
            dtrain = xgb.DMatrix(df[list(targets)], label=df[y])
            model = xgb.train(parameter_copy, dtrain, num_rounds)
            name_remove = [remove_continuous_discrete_prefix(x) for x in targets]
            name = "discrete_" + "_".join(name_remove)

            self.trans_ls.append(('xgb', name, targets, model, pos))

    def _fit_lightgbm(self, df, y, targets_list, parameter):
        for targets in targets_list:
            parameter_copy = copy.deepcopy(parameter)
            if 'num_threads' not in parameter.keys():
                parameter_copy['num_threads'] = self.nthread
            if 'objective' not in parameter.keys():
                if len(np.unique(df[y])) == 2:
                    parameter_copy['objective'] = "binary"
                else:
                    parameter_copy['objective'] = "multiclass"
            num_rounds = parameter_copy['num_threads']
            pos = parameter_copy['pos']
            parameter_copy.pop("pos")
            dtrain = lgb.Dataset(df[list(targets)], label=df[y])
            model = lgb.train(parameter_copy, dtrain, num_rounds)

            name_remove = [remove_continuous_discrete_prefix(x) for x in targets]
            name = "discrete_" + "_".join(name_remove)
            self.trans_ls.append(('lgb', name, targets, model, pos))

    def transform(self, df):
        result = df.copy(deep=True)
        trans_results = [result]
        for method, name, targets, model, pos in self.trans_ls:
            if method == 'xgb':
                tree_infos: pd.DataFrame = model.trees_to_dataframe()
            elif method == 'lgb':
                tree_infos = tree_to_dataframe_for_lightgbm(model).get()
            else:
                raise Exception("Not Implemented Yet")

            trans_results.append(self._boost_transform(result[list(targets)], method, name, pos, tree_infos))
        return pd.concat(trans_results, axis=1)

    @staticmethod
    def _transform_byeval(row, leaf_condition):
        for key in leaf_condition.keys():
            if eval(leaf_condition[key]):
                return key
        return np.NaN

    def _boost_transform(self, df, method, name, pos, tree_infos):
        tree_ids = tree_infos["Node"].drop_duplicates().tolist()
        tree_ids.sort()
        for tree_id in tree_ids:
            tree_info = tree_infos[tree_infos["Tree"] == tree_id][
                ["Node", "Feature", "Split", "Yes", "No", "Missing"]].copy(deep=True)
            tree_info["Yes"] = tree_info["Yes"].apply(lambda y: str(y).replace(str(tree_id) + "-", ""))
            tree_info["No"] = tree_info["No"].apply(lambda y: str(y).replace(str(tree_id) + "-", ""))
            tree_info["Missing"] = tree_info["Missing"].apply(lambda y: str(y).replace(str(tree_id) + "-", ""))
            leaf_nodes = tree_info[tree_info["Feature"] == "Leaf"]["Node"].drop_duplicates().tolist()
            encoder_dict = {}
            for leaf_node in leaf_nodes:
                encoder_dict[leaf_node] = get_booster_leaf_condition(leaf_node, [], tree_info)
            if not encoder_dict:
                continue
            df.fillna(np.NaN)
            feature_name = "_".join([name, method, "tree_" + str(tree_id), pos])
            df.columns = [str(col) for col in list(df.columns)]
            add_feature = pd.DataFrame(df.apply(self._transform_byeval, leaf_condition=encoder_dict, axis=1),
                                       columns=[feature_name])
            df = pd.concat([df, add_feature], axis=1)

        return df


class AnomalyScoreEncoder(EncoderBase):
    def __init__(self, nthread=None):
        super(AnomalyScoreEncoder, self).__init__()
        if nthread:
            self.nthread = cpu_count
        else:
            self.nthread = nthread

    def fit(self, df, y, targets_list, config):
        self.reset()
        for method, parameter in config:
            if method == 'IsolationForest':
                self._fit_isolationForest(df, y, targets_list, parameter)
            if method == 'LOF':
                self._fit_LOF(df, y, targets_list, parameter)

    def transform(self, df):
        result = df.copy(deep=True)
        for method, name, targets, model in self.trans_ls:
            result[name + "_" + method] = model.predict(df[targets])

        return result

    def _fit_isolationForest(self, df, y, targets_list, parameter):
        for targets in targets_list:
            n_jobs = self.nthread

            model = IsolationForest(n_jobs=n_jobs)
            model.fit(X=df[targets])

            name_remove = [remove_continuous_discrete_prefix(x) for x in targets]
            name = "discrete_" + "_".join(name_remove)
            self.trans_ls.append(('IsolationForest', name, targets, model))

    def _fit_LOF(self, df, y, targets_list, parameter):
        for targets in targets_list:
            n_jobs = self.nthread

            model = LocalOutlierFactor(n_jobs=n_jobs)
            model.fit(X=df[targets])

            name_remove = [remove_continuous_discrete_prefix(x) for x in targets]
            name = "discrete_" + "_".join(name_remove)
            self.trans_ls.append(("LOF", name, targets, model))


class GroupbyEncoder(EncoderBase):  # TODO: Not Finished Yet
    def __init__(self):
        super(GroupbyEncoder, self).__init__()

    def fit(self, df, targets, groupby_op_list):
        self.reset()
        for target in targets:
            for groupby, operations, param in groupby_op_list:
                for operation in operations:
                    groupby_result = self._fit_one(df, target, groupby, operation)
                    name = target + '_groupby_' + '_'.join(groupby) + '_op_' + operation
                    groupby_result = groupby_result.rename(columns={target: name})
                    self.trans_ls.append((groupby, groupby_result))

    def transform(self, df):
        result = df.copy(deep=True)
        for groupby, groupby_result in self.trans_ls:
            result = result.merge(groupby_result, on=groupby, how='left')
        return result

    def _fit_one(self, df, target, groupby_vars, operation):
        result = df.groupby(groupby_vars, as_index=False).agg({target: operation})
        return result

# Deprecated
# class TargetMeanEncoder(object):
#     """
#     This is basically a duplicate
#     """
#
#     def __init__(self, smoothing_coefficients=None):
#         warnings.warn("This is deprecated!Please do not use this anymore.")
#         if not smoothing_coefficients:
#             self.smoothing_coefficients = [1]
#         else:
#             self.smoothing_coefficients = smoothing_coefficients
#
#     def fit_and_transform_train(self, df_train, ys, target_vars, n_splits=5):
#         splitted_df = split_df(df_train, n_splits=n_splits, shuffle=True)
#         result = list()
#         for train_df, test_df in splitted_df:
#             for y in ys:
#                 for target_var in target_vars:
#                     for smoothing_coefficient in self.smoothing_coefficients:
#                         test_df = self._fit_one(train_df, test_df, y, target_var, smoothing_coefficient)
#             result.append(test_df)
#         return pd.concat(result)
#
#     def _fit_one(self, train_df, test_df, y, target_var, smoothing_coefficient):
#         global_average = train_df[y].mean()
#         local_average = train_df.groupby(target_var)[y].mean().to_frame().reset_index()
#         name = "target_mean_" + y + "_" + target_var + "_lambda_" + str(smoothing_coefficient)
#         local_average = local_average.rename(columns={y: name})
#         test_df = test_df.merge(local_average, on=target_var, how='left')
#         test_df[name] = test_df[name].map(
#             lambda x: global_average if pd.isnull(x) else smoothing_coefficient * x + (
#                     1 - smoothing_coefficient) * global_average)
#         return test_df


def get_interval(x, sorted_intervals):
    if pd.isnull(x):
        return -1
    if x == np.inf:
        return -2
    if x == -np.inf:
        return -3
    interval = 0
    found = False
    sorted_intervals.append(np.inf)
    if x < sorted_intervals[0] or x >= sorted_intervals[len(sorted_intervals)-1]:
        return -4
    while not found and interval < len(sorted_intervals) - 1:
        if sorted_intervals[interval] <= x < sorted_intervals[interval + 1]:
            return interval
        else:
            interval += 1


def encode_label(x):
    x_copy = x.copy(deep=True)
    unique = sorted(list(set([str(item) for item in x_copy.astype(str).unique()])))
    kv = {unique[i]: i for i in range(len(unique))}
    x_copy = x_copy.map(lambda x: kv[str(x)])
    return x_copy


def get_uniform_interval(minimum, maximum, nbins):
    result = [minimum]
    step_size = (float(maximum - minimum)) / nbins
    for index in range(nbins - 1):
        result.append(minimum + step_size * (index + 1))
    result.append(maximum)
    return result


def get_quantile_interval(data, nbins):
    quantiles = get_uniform_interval(0, 1, nbins)
    return list(data.quantile(quantiles))


def to_str(x):
    if pd.isnull(x):
        return '#NA#'
    else:
        return str(x)

def get_booster_leaf_condition(leaf_node, conditions, tree_info: pd.DataFrame):
    start_node_info = tree_info[tree_info["Node"] == leaf_node]
    if start_node_info["Feature"].tolist()[0] == "Leaf":
        conditions = []

    if str(leaf_node) in tree_info["Yes"].drop_duplicates().tolist():
        father_node_info = tree_info[tree_info["Yes"] == str(leaf_node)]
        fathers_left = True
    else:
        father_node_info = tree_info[tree_info["No"] == str(leaf_node)]
        fathers_left = False

    father_node_id = father_node_info["Node"].tolist()[0]
    split_value = father_node_info["Split"].tolist()[0]
    split_feature = father_node_info["Feature"].tolist()[0]

    if fathers_left:
        add_condition = ["row['" + str(split_feature) + "'] <= " + str(split_value)]
        if father_node_info["Yes"].tolist()[0] == father_node_info["Missing"].tolist()[0]:
            add_condition.append("is_missing(row['" + str(split_feature) + "'])")

    else:
        add_condition = ["row['" + str(split_feature) + "'] > " + str(split_value)]
        if father_node_info["No"].tolist()[0] == father_node_info["Missing"].tolist()[0]:
            add_condition.append("row['" + str(split_feature) + "'] == np.NaN")
    add_condition = "(" + " or ".join(add_condition) + ")"
    conditions.append(add_condition)

    if father_node_info["Node"].tolist()[0] == 0:
        return " and ".join(conditions)
    else:
        return get_booster_leaf_condition(father_node_id, conditions, tree_info)


class tree_to_dataframe_for_lightgbm(object):
    def __init__(self, model):
        self.json_model = model.dump_model()
        self.features = self.json_model["feature_names"]

    def get_root_nodes_count(self, tree, max_id):
        tree_node_id = tree.get("split_index")
        if tree_node_id:
            if tree_node_id > max_id:
                max_id = tree_node_id

        if tree.get("left_child"):
            left = self.get_root_nodes_count(tree.get("left_child"), max_id)
            if left > max_id:
                max_id = left
        else:
            left = []

        if tree.get("right_child"):
            right = self.get_root_nodes_count(tree.get("right_child"), max_id)
            if right > max_id:
                max_id = right
        else:
            right = []

        if not left and not right:  # 如果root是叶子结点
            max_id = max_id
        return max_id

    def get(self):
        tree_dataframe = []
        for tree in self.json_model["tree_info"]:
            tree_id = tree["tree_index"]
            tree = tree["tree_structure"]
            root_nodes_count = self.get_root_nodes_count(tree, 0) + 1
            tree_df = self._lightGBM_trans(tree, pd.DataFrame(), tree_id, root_nodes_count).sort_values(
                "Node").reset_index(drop=True)
            tree_df["Tree"] = tree_id
            tree_dataframe.append(tree_df)

        return pd.concat(tree_dataframe, axis=0)

    def _lightGBM_trans(self, tree, tree_dataFrame, tree_id, root_nodes_count):
        tree_node_id = tree.get("split_index")
        threshold = tree.get("threshold")
        default_left = tree.get("default_left")

        if tree_node_id is not None:
            data = {"Node": tree_node_id, "Feature": self.features[tree.get("split_feature")], "Split": threshold}
            yes_id = tree.get("left_child").get("split_index")
            if yes_id is None:
                yes_id = tree.get("left_child").get("leaf_index") + root_nodes_count
            tree_dataFrame = self._lightGBM_trans(tree.get("left_child"), tree_dataFrame, tree_id, root_nodes_count)

            no_id = tree.get("right_child").get("split_index")
            if no_id is None:
                no_id = tree.get("right_child").get("leaf_index") + root_nodes_count

            tree_dataFrame = self._lightGBM_trans(tree.get("right_child"), tree_dataFrame, tree_id, root_nodes_count)

            if default_left:
                missing_id = yes_id
            else:
                missing_id = no_id

            data["Yes"], data["No"], data["Missing"] = "_".join([str(yes_id)]), "_".join(
                [str(no_id)]), "_".join([str(missing_id)])
        else:
            data = {"Node": root_nodes_count + tree.get("leaf_index"), "Feature": "Leaf", "Split": None, "Yes": None,
                    "No": None, "Missing": None}

        row = pd.DataFrame.from_dict(data, orient="index").T
        tree_dataFrame = pd.concat([tree_dataFrame, row])
        return tree_dataFrame


class StandardizeEncoder(EncoderBase):
    def __init__(self):
        super(StandardizeEncoder, self).__init__()

    def fit(self, df, targets, ):
        self.reset()
        for target in targets:
            mean = df[target].mean()
            std = df[target].std()
            new_name = 'continuous_' + remove_continuous_discrete_prefix(target) + '_standardized'
            self.trans_ls.append((target, mean, std, new_name))

    def transform(self, df):
        result = df.copy(deep=True)
        for target, mean, std, new_name in self.trans_ls:
            result[new_name] = (result[target] - mean) / std
        return result


class InteractionEncoder:
    def __init__(self):
        self.level = list()
        self.targets = None

    def fit(self, targets, level='all'):
        if level == 'all':
            self.level = [2, 3, 4]
        else:
            self.level = level
        self.targets = targets

    def transform(self, df):
        result = df.copy(deep=True)
        for level in self.level:
            if level == 2:
                for target_1 in self.targets:
                    for target_2 in self.targets:
                        new_name = 'continuous_' + remove_continuous_discrete_prefix(
                            target_1) + "_" + remove_continuous_discrete_prefix(target_2) + "_cross"
                        result[new_name] = result[target_1] * result[target_2]
            if level == 3:
                for target_1 in self.targets:
                    for target_2 in self.targets:
                        for target_3 in self.targets:
                            new_name = 'continuous_' + remove_continuous_discrete_prefix(
                                target_1) + "_" + remove_continuous_discrete_prefix(
                                target_2) + "_" + remove_continuous_discrete_prefix(target_3) + "_cross"

                            result[new_name] = result[target_1] * result[target_2] * result[target_3]
            if level == 4:
                for target_1 in self.targets:
                    for target_2 in self.targets:
                        for target_3 in self.targets:
                            for target_4 in self.targets:
                                new_name = 'continuous_' + remove_continuous_discrete_prefix(
                                    target_1) + "_" + remove_continuous_discrete_prefix(
                                    target_2) + "_" + remove_continuous_discrete_prefix(
                                    target_3) + "_" + remove_continuous_discrete_prefix(target_4) + "_cross"
                                result[new_name] = result[target_1] * result[target_2] * result[target_3] * result[
                                    target_4]
        return result


class DimReducEncoder:
    def __init__(self):
        self.result = list()

    def fit(self, df, targets, config):
        for target in targets:
            for method, parameter in config:
                if method == 'pca':
                    n_comp = parameter['n_components']
                    pos = parameter.get('pos') if parameter.get("pos") else ""
                    encoder = PCA(n_comp)
                    encoder.fit(df[target])
                    self.result.append((method, encoder, pos, n_comp, target))
                if method == 'tsne':
                    if parameter.get("pos"):
                        pos = parameter.get("pos")
                        parameter.pop("pos")
                    else:
                        pos = ""
                    encoder = TSNE(**parameter)
                    encoder.fit(df[target])
                    self.result.append((method, encoder, pos, parameter.get("n_components"), target))

    def transform(self, df):
        result = df.copy(deep=True)
        for method, encoder, pos, n_comp, target in self.result:
            if method == 'pca':
                new_names = ["pca_" + str(x) + pos for x in range(n_comp)]
                result = pd.concat([result, pd.DataFrame(encoder.transform(df[target]), columns=new_names)], axis=1)

            elif method == "tsne":
                new_names = ["tsne_" + str(x) + pos for x in range(n_comp)]
                result = pd.concat([result, pd.DataFrame(encoder.embedding_, columns=new_names)], axis=1)

        return result


def is_missing(v):
    return v == np.NaN
