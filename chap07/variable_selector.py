# coding = 'utf-8'
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import ray
import numpy as np
from ind_cols import get_ind_col
from .model_fitter import LGBFitter, XGBFitter, CATFitter
import copy
import shap
import statsmodels.api as sm


def permutate_selector(train_df, eval_df, y, variables=None, metric='acc', **kwargs):  # TODO Add more metric

    """
    Return the importance of variables based on permutation loss

    :param train_df: training data set
    :param eval_df: eval data set
    :param y: name of the target variable
    :param variables: the variables to select perform the select; if None, then all the variables except target variable will be selected
    :param metric: the metric to determine the order; higher value indicate better performance
    :param **kwargs: argument for logistic regression

    returns: result after permutation
    """

    @ray.remote()
    def fit_and_predict(train_df, eval_df, y, variables, metric, start=None, **kwargs):
        if start is None:
            clf = LogisticRegression(**kwargs)
        else:
            clf = LogisticRegression(warm_start=start, **kwargs)
        clf.fit(train_df[variables], train_df[y])
        y_pred = clf.predict(eval_df[variables])

        if metric == 'acc':  # TODO Add more metric
            score = accuracy_score(eval_df[y], y_pred)
        else:
            score = None
        return score, clf.coef_

    @ray.remote()
    def fit_permute_and_predict(train_df, eval_df, y, variables, metric, start, permute_var, **kwargs):
        train_df[permute_var] = np.random.permutation(train_df[permute_var])
        score, _ = fit_and_predict(train_df, eval_df, y, variables, metric, start, **kwargs)
        return (permute_var, score)

    ray.init()

    ind_col = get_ind_col(train_df)
    if variables is not None:
        var_to_use = [x for x in ind_col if x in variables]
    else:
        var_to_use = [x for x in ind_col if x != y]

    result_dict = dict()

    score, warm_start = fit_and_predict(train_df, eval_df, y, var_to_use, metric, None, **kwargs)
    result_dict['origin'] = score

    train_df_id = ray.put(train_df)
    eval_df_id = ray.put(eval_df)

    var_to_use_id = ray.put(var_to_use)
    start_id = ray.put(warm_start)
    result = [
        fit_permute_and_predict.remote(train_df_id, eval_df_id, y, var_to_use_id, start_id, permute_var, **kwargs, ) for
        permute_var in var_to_use]
    result_list = ray.get(result)

    for var, score in result_list:
        result_dict[var] = score
    ray.shutdown()
    return result_dict


def tree_selector(train_df, eval_df, y, opt, metric="error", type='lgb'):
    """
    This function select variable importance using built functions from xgboost or lightgbm
    :param train_df: training dataset,
    :param eval_df: evaluation dataset
    :param y: target variable
    :param opt: training operation for tree models
    :param metric: the metric used to select the best number of trees; currently only support 'error'
    :param type: 'lgb' or 'xgb'
    """

    if type == 'lgb':
        trainer = LGBFitter(y, metric)
    elif type == 'xgb':
        trainer = XGBFitter(y, metric)
    else:
        raise NotImplementedError()
    trainer.train(train_df, eval_df, opt)
    best_round = trainer.best_round
    opt_copy = copy.deepcopy(opt)
    opt_copy['num_round'] = best_round
    trainer.train(train_df, eval_df, opt_copy)
    importance = trainer.clf.feature_importance_
    name = train_df.drop(columns=y).columns

    result_dict = dict()
    for k, v in zip(name, importance):
        result_dict[k] = v
    return result_dict


def shap_selector(train_df, eval_df, y, opt, type='lgb', metric='error'):
    """
    This returns the shap explainer so that one can use it for variable selection.
    The base tree model we use will select the best iterations
    :param train_df: training dataset
    :param eval_df: eval dataset
    :param y: the target variable name
    :param opt: training argument for boosting parameters
    :param type; 'lgb' , 'xgb' or 'catboost'. The tree used for computing shap values.
    :param metric: metric to select the best tree; currently only support 'error'
    :returns shap explainer
    """
    opt_copy = copy.deepcopy(opt)
    if type == 'lgb':
        trainer = LGBFitter(y, metric)

    elif type == 'xgb':
        trainer = XGBFitter(y, metric)
    elif type == 'catboost':
        trainer = CATFitter(y, metric)
    else:
        raise NotImplementedError()
    trainer.train(train_df, eval_df, opt_copy)
    best_round = trainer.best_round
    opt_copy['num_round'] = best_round
    trainer.train(train_df, eval_df, opt_copy)
    clf = trainer.clf
    return shap.TreeExplainer(clf)


def vif_selector(data_df, y):
    """
    Calculate the VIF to select variables.
    :param data_df: the dataset
    :param y: the target variable
    """

    ray.init()

    ex_var = [var for var in data_df.columns if var != y]

    data_df_id = ray.put(data_df)

    @ray.remote()
    def ls(data_df, target_var):
        model = sm.OLS(data_df.drop(columns=target_var), data_df[target_var])
        result = model.fit()
        rs = result.rsquare
        if rs == 1:
            vif = np.inf
        else:
            vif = 1.0 / (1.0 - rs)

        return (target_var, vif)

    result = [ls.remote(data_df_id, target_var, ) for target_var in ex_var]
    result_list = ray.get(result)

    return_result = dict()

    for k, v in result_list:
        return_result[k] = v
    ray.shutdown()
    return return_result
