import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
import random
import json
import os
from catboost import CatBoostRegressor
import numpy as np
import random
from utils import *
from fastai.tabular import *

VALID_TIME = 66225 #Time at which after this point is the final 14 days of the given dataset
locs = ['geohash6', 'cluster_l', 'cluster_s']
periods = ['timestamp', 'day', 'hr']
agg_fns = ['mean', 'median', 'max', 'min', 'std']

other_cols = ['geohash6', 'day', 'index', 'time']
base_cols = ['demand',  'hr', 'lat', 'lon', 'min']
cluster_cols = ['cluster_s', 'cluster_l', 'cs_demand', 'cl_demand']
dl_t_cols = [f't{i}' for i in range(-5, 0)]
dl_cl_cols = [f'cl{i}' for i in range(-5, 1)]
dl_cs_cols = [f'cs{i}' for i in range(-5, 1)]
t_cols = [f't{i}' for i in range(-96, 0)]
cl_cols = [f'cl{i}' for i in range(-96, 1)]
cs_cols = [f'cs{i}' for i in range(-96, 1)]
stat_cols = [f'{loc}_{period}_{agg}' for loc in locs for period in periods for agg in agg_fns]
holt_t_cols = [f'holt_t_{i}' for i in range(-5, 6)]
holt_l_cols = [f'holt_l_{i}' for i in range(-5, 6)]
holt_s_cols = [f'holt_s_{i}' for i in range(-5, 6)]

steps = [base_cols, cluster_cols, t_cols, cl_cols, cs_cols, stat_cols, holt_t_cols, holt_l_cols, holt_s_cols]
dl_steps = [base_cols, cluster_cols, dl_t_cols, dl_cl_cols, dl_cs_cols, stat_cols, holt_t_cols, holt_l_cols, holt_s_cols]
step_fns = ['base_cols', 'cluster_cols', 't_cols', 'cl_cols', 'cs_cols', 'stat_cols', 'holt_t_cols', 'holt_l_cols', 'holt_s_cols']

path = Path(os.path.abspath('./')+'models')
all_features = [i for step in steps for i in step]
dl_features = [i for step in steps for i in dl_steps]

def train(fn='df_train_large.p', cols=all_features, task_type='GPU'):
    best_params_1 = load_pickle('best_params_1.p')
    best_params_2 = load_pickle('best_params_2.p')
    best_params_3 = load_pickle('best_params_3.p')
    best_params_4 = load_pickle('best_params_4.p')
    best_params_5 = load_pickle('best_params_5.p')
    best_params = [None, best_params_1, best_params_2, best_params_3, best_params_4, best_params_5]
    df_train = load_pickle(fn)
    print(f"No. of columns = {len(df_train.columns)}")
    models = {}

    for t in range(1,6):

        models[t] = {}
        print(f"Doing pred for time {t}")
        df = df_train.loc[df_train[f't{t}'].notna()]
        print(f"Number of columns in df = {len(df.columns)}")
        train = df.loc[df['time']<=VALID_TIME]
        test = df.loc[df['time']>VALID_TIME]
        for model_id in range(0,5):
            print(f"Doing model {model_id}")
            params = best_params[t][model_id][1]
            params['depth'] += 1
            params['iterations'] += random.randint(2000, 3000)
            #print(f"---Param depth = {params['depth']}---")
            #print(f"---Param iteration = {params['iterations']}---")
            clf = CatBoostRegressor(**params, task_type=task_type)
            print(f"Shape of training data = {train[cols].values.shape}")
            clf.fit(train[cols].values, np.ravel(train[f't{t}'].values), cat_features=None)
            train_score = rmse_(clf.predict(train[cols].values), np.ravel(train[f't{t}'].values))
            test_score = rmse_(clf.predict(test[cols].values), np.ravel(test[f't{t}'].values))
            print(f"Score for time {t} model {model_id}= Train: {train_score} Test: {test_score}")
            # Train on full dataset
            clf_final = CatBoostRegressor(**params, task_type=task_type)
            #print(f"Shape of final data = {df[cols].values.shape}")
            clf_final.fit(df[cols].values, np.ravel(df[f't{t}'].values), cat_features=None)
            models[t][model_id] = clf_final
            clf_final.save_model(f'models/cb_{t}_{model_id}')
            #save_pickle(models, model_fn)
            #save_pickle(scores, scores_fn)
    return models

def average_predict(models, fn='df_train.p', cols=all_features):
    df_train = load_pickle(fn)
    scores = []
    for t in models:
        df = df_train.loc[df_train[f't{t}'].notna()]
        print(f"Number of columns in df = {len(df.columns)}")
        train = df.loc[df['time']<=VALID_TIME]
        test = df.loc[df['time']>VALID_TIME]
        preds = []
        for model_id in models[t]:
            clf = models[t][model_id]
            test_preds = clf.predict(test[cols].values)
            preds.append(test_preds)
            test[f"preds_{model_id}"] = test_preds
            train_preds = clf.predict(train[cols].values)
            train[f"preds_{model_id}"] = train_preds

        save_pickle(train, f"meta_train_{t}.p")
        save_pickle(test, f"meta_test_{t}.p")
        final = np.mean(preds, axis=0)
        score = rmse_(final, np.ravel(test[f't{t}'].values))
        print(f"Average score for t{t} = {score}")
        scores.append(score)
    return scores

def train_meta(models, mean_scores, cols=all_features, task_type='GPU'):
    best_params_1 = load_pickle('best_params_1.p')
    best_params_2 = load_pickle('best_params_2.p')
    best_params_3 = load_pickle('best_params_3.p')
    best_params_4 = load_pickle('best_params_4.p')
    best_params_5 = load_pickle('best_params_5.p')
    best_params = [None, best_params_1, best_params_2, best_params_3, best_params_4, best_params_5]
    meta_models = {}
    for time in range(1,6):
        train = load_pickle(f"meta_train_{time}.p")
        test = load_pickle(f"meta_test_{time}.p")

        print(f"RMSE using mean = {mean_scores[time-1]}")
        print(f"Building Meta model cb {time}")
        # train = train.fillna(train.mean())
        # test = test.fillna(test.mean())

        meta_cols = cols + [f"preds_{model_id}" for model_id in models[time]]
        #ridge = Ridge(alpha=2e-1).fit(train[ridge_cols].values, np.ravel(train[f't{time}'].values))
        #ridge_preds = ridge.predict(test[ridge_cols].values)
        params = best_params[time][0][1]
        params['depth'] += 1
        params['iterations'] += random.randint(2000, 4000)
        meta_clf = CatBoostRegressor(**params, task_type=task_type).fit(train[meta_cols].values, 
                                                                        np.ravel(train[f't{time}'].values), 
                                                                        cat_features=None)
        meta_preds = meta_clf.predict(test[meta_cols].values)
        meta_rmse = rmse_(meta_preds, np.ravel(test[f't{time}'].values))
        print(f"RMSE using meta_learner = {meta_rmse}")
        meta_models[time] = meta_clf
    return meta_models


#==================Deep learning==========================

# cat_vars = ['geohash6', 'hr', 'min', 'timestamp']
# cont_vars = [var for var in dl_features if var not in cat_vars]
procs=[FillMissing, Categorify, Normalize]
bs = 1024
y_range = torch.tensor([0,1], device=defaults.device)

def build_one_dataset(df_full, t, path, cat_vars, cont_vars, procs):
    '''
    Build one dataset for training. Difference between this and build_final_dataset
    is that build_final_dataset uses the entire data as the training set
    while this function splits it into a training and validation set.
    The validation set here is defined as the last 2 weeks of the 
    '''
    cat_vars = ['geohash6', 'hr', 'min', 'timestamp']
    cont_vars = [var for var in dl_features if var not in cat_vars] 
    df = df_full.loc[df_full[f't{t}'].notna()].copy()[cat_vars + cont_vars + [f't{t}', 'time']]
    valid_idx = list(range(len(df.loc[df['time']<=VALID_TIME]), len(df)))
    # test_df = df.loc[df['time']>VALID_TIME]
    dep_var = f't{t}'
    data = (TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs,)
                .split_by_idx(valid_idx)
                .label_from_df(cols=dep_var, label_cls=FloatList)
    #           .add_test(TabularList.from_df(test_df, path=path, cat_names=cat_vars, cont_names=cont_vars))
                .databunch(bs=bs))
    return data

def build_final_dataset(df_full, t, path, cat_vars, cont_vars, procs):
    '''
    Build final dataset for model to be deployed
    '''
    cat_vars = ['geohash6', 'hr', 'min', 'timestamp']
    cont_vars = [var for var in dl_features if var not in cat_vars] 
    df = df_full.loc[df_full[f't{t}'].notna()].copy()[cat_vars + cont_vars + [f't{t}', 'time']]
    dep_var = f't{t}'
    data = (TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs,)
                .split_none()
                .label_from_df(cols=dep_var, label_cls=FloatList)
#                .add_test(TabularList.from_df(test_df, path=path, cat_names=cat_vars, cont_names=cont_vars))
                .databunch(bs=1024))
    return data

def build_data(cat_vars, cont_vars, fn='df_train.p'):
    '''
    Build list of datasets corresponding to the 5 time intervals for training
    '''
    df_train = load_pickle(fn)
    df_train.reset_index(inplace=True)
    df_train.sort_values(['time'], inplace=True)
    data = [build_one_dataset(df_train, t, path, cat_vars, cont_vars, procs) for t in range(1,6)]
    return data

def build_models(data, y_range, prefix, layers=[1000,500], ps=[0.001,0.01], emb_drop=0.04, cycle=5, lr=3e-2, wd=0.2):
    models = {}
    for n, dataset in enumerate(data):
        learn = tabular_learner(dataset, layers=layers, ps=ps, emb_drop=emb_drop, 
                        y_range=y_range, metrics=root_mean_squared_error)
        learn.fit_one_cycle(cycle, lr, wd=wd)
        models[n+1] = learn
        learn.save(f"{prefix}{n+1}")
        learn.export(f"{prefix}{n+1}.pkl")
    return models

def load_models(data, prefix='trained_model5e_', layers=[1000,500], ps=[0.01,0.01], emb_drop=0.04):
    models = {}
    for n in range(0,5):
        #learn = load_learner(path, f"trained_model5e_{n+1}.pkl")
        learn = tabular_learner(data[n], layers=[1000,500], ps=[0.001,0.01], emb_drop=0.04, 
                y_range=y_range, metrics=root_mean_squared_error)
        learn = learn.load(f"{prefix}{n+1}")
        models[n+1] = learn
    return models

def finetune(data, output, prefix='final_model_v1', layers=[1000,500], ps=[0.05,0.05], emb_drop=0.04, cycle=5, lr=3e-3, wd=0.2):
    models = load_models(data, prefix=prefix, layers=layers, ps=ps, emb_drop=emb_drop)
    for model_id in models:
        learn = models[model_id]
        learn.fit_one_cycle(cycle, lr, wd=wd)
        learn.save(f"{output}{model_id}")
        learn.export(f"{output}{model_id}.pkl")

def unique_cats(df_train):
    res = {}
    for t in range(1,6):
        res[t] = {}
        df = df_train.loc[df_train[f't{t}'].notna()]
        res[t]['geohash6'] = set(df['geohash6'].unique().tolist())
        res[t]['timestamp'] = set(df['timestamp'].unique().tolist())
        res[t]['hr'] = set(df['hr'].unique().tolist())
        res[t]['min'] = set(df['min'].unique().tolist())
        res[t]['geohash6_mode'] = df['geohash6'].mode()[0]
        res[t]['timestamp'] = df['timestamp'].mode()[0]
        res[t]['hr'] = df['hr'].mode()[0]
        res[t]['min'] = df['min'].mode()[0]
    save_pickle(res, 'unique_cats.p')
    return res



