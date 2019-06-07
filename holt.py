# Holt preprocessing
from utils import *
import json
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
import pickle
import datetime
import random
import os

DIR = 'templates/'
HOLT_DIR = 'holt/'
PERIOD = 20145

def get_grp_indices(loc_index_d, grp):
    """
    Get indices for 
    """
    if grp in loc_index_d:
        return listify(loc_index_d[grp])
    raise KeyError(f'Group {grp} not found')


def pred_holt(df, holt_params, grp, idx, type_, processed, failed_holt, test=False):
    grp = str(grp)
    params = None
    if f'{grp}-{idx}' in processed:
        # Write to disk
        with open(f'{HOLT_DIR}{type_}/{grp}-{idx}.txt', 'r') as f:
            d = json.load(f)
        inner_d = d[list(d.keys())[0]]
        idx = list(inner_d.keys())[0]
        #print(idx)
        #print(inner_d[str(idx)])
        pred_values = np.array([float(x) for x in inner_d[idx][1:-1].split(' ') if x.strip()])
        #print((idx, pred_values))
        return (int(idx), pred_values)

    try:
        df_sliced = df.loc[(df['time']>=(idx-PERIOD)) & (df['time']<=idx)].copy()
        df_sliced_processed = process_grp_df(df_sliced)
        train = df_sliced_processed['demand'].copy()
    except TypeError as e:
        print(e)
        print(f"Idx that failed = {idx} from grp {grp}")
        if not failed_holt.get(type_):
            failed_holt[type_] = {}
        if not failed_holt[type_].get(grp):
            failed_holt[type_][grp] = []
        failed_holt[type_][grp].append(idx)
        return (idx, None)

    try:
#         with open(f'{DIR}{type_}/{grp}.txt', 'r') as f:
#             params = json.load(f)
        params = holt_params[type_][grp]
    except Exception as e:
        print(e)
        print(f"Cannot find params of {type_}-{grp}")
        if not test:
            
            #print(f'Failed to load {DIR}{type_}/{grp}.txt')
            if not failed_holt.get(type_):
                failed_holt[type_] = {}
            if not failed_holt[type_].get(grp):
                failed_holt[type_][grp] = []
            failed_holt[type_][grp].append(idx)
            return (idx, None)
        pass
            
    #print(params)
    #print(train)
    try:
        if not params and not test:
            print(f"Unable to build holt for {grp} {idx}")
            return (idx, None)
        if params:
            model = ExponentialSmoothing(train, 
                                         seasonal='mul', 
                                         seasonal_periods=96).fit(smoothing_level=params['smoothing_level'], 
                                                                  smoothing_slope=params['smoothing_slope'], 
                                                                  smoothing_seasonal=params['smoothing_seasonal'], 
                                                                  optimized=False)
        elif test:
            print("Building model from scratch")
            model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=96).fit()
        #print(model.params)
        pred = model.predict(start=train.index[-1]-datetime.timedelta(minutes=15*5), end=train.index[-1]+datetime.timedelta(minutes=15*5))
        #print(pred.values)
        
        # Write to disk
        with open(f'{HOLT_DIR}{type_}/{grp}-{idx}.txt', 'w') as f:
            json.dump({str(grp):{str(idx):str(pred.values)}}, f)
        
        return (idx, pred.values)
    except:
        print(f"Length of train for {grp}: {len(train)}")
        if not failed_holt.get(type_):
            failed_holt[type_] = {}
        if not failed_holt[type_].get(grp):
            failed_holt[type_][grp] = []
        failed_holt[type_][grp].append(idx)
        return (idx, None)
        
        
def process_grp_df(df):
    # Get initial mean of demand
    grp_mean = df['demand'].mean()
    # Set index to dt
    df['dt'] = df['time'].apply(lambda x: time_to_dt(x))
    df = df.set_index('dt')
#     print(df.index.is_unique)
#     print(df.head(20))
    df = df.sort_values('time')
    df = df.asfreq('15T')
    
    # Fill missing values
    memo = {}
    for n, row in df[df['demand'].isnull()].iterrows():
        try:
            # Fill missing value with the average for that time slot
            hh = str(n.hour)
            mm = str(n.minute)
            if not memo.get(f'{hh}:{mm}'):
                candidate = df['demand'][df['timestamp']==f'{hh}:{mm}'].mean()
                if pd.notna(candidate):
                    memo[f'{hh}:{mm}'] = candidate
                else:
                    memo[f'{hh}:{mm}'] = grp_mean
            #print(f"memo mean = {memo[f'{hh}:{mm}']}")
            df.set_value(n, 'demand', memo[f'{hh}:{mm}'])
        except:
            #print(e)
            # Fill missing value with the average for that grp
            df.set_value(n, 'demand', grp_mean)
    return df
        

def holt_sample(df, holt_params, grp, indices, type_, processed, failed_holt, parallel=True, test=False):
    res = None
    if parallel:
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(pred_holt)(df, holt_params, grp, idx, type_, processed, failed_holt, test=test) for idx in indices)
        res = executor(tasks)
    else: 
        #print(indices)
        res = [pred_holt(df, holt_params, grp, idx, type_, processed, failed_holt, test=test) for idx in indices]
    return res

def get_holt_for_grp(grps, holt_params, df, col, loc_index_d, failed_grp, failed_holt, parallel=False, test=False):
    grp_d = {}
    type_ = 'locs' if col == 'geohash6' else 'l' if col == 'cluster_l' else 's'
    
    save_dir = f'{HOLT_DIR}{type_}/'
    ensure_dir(save_dir)
        
    processed = set([x.split('.')[0] for x in os.listdir(save_dir)])
    for n, grp in enumerate(grps):
        print(f"Doing item {n}, {grp}")
        if grp in failed_grp:
            if not test:
                print(f"Skipping failed grp {grp})")
                continue
        indices = get_grp_indices(loc_index_d, grp) # in time(int) format
        #print(indices)
        if not indices:
            continue
        
        col = col if col == 'geohash6' else 'cluster'
        res = holt_sample(df[df[col]==grp].copy(), holt_params, grp, indices, type_, processed, failed_holt, parallel=parallel, test=test)
        #print(res)
        grp_d[grp] = {idx:pred for (idx, pred) in res}
        #break
    return grp_d
 
#===============MAIN FUNCTION HERE==================
def get_holt(raw_df, holt_params, loc_index_d, failed_grp, col='geohash6', parallel=False, test=False):
    #dfs should be in the same order as cols
    # Extract grp for every grp to be processed
    grps = list(raw_df[col].unique())
    res = {}
    failed_holt = {}
    print(f"Doing col {col}")
    grp_d = get_holt_for_grp(grps, holt_params, raw_df, col, loc_index_d, failed_grp, failed_holt, parallel=parallel, test=test)
    res[col]=grp_d
    
    return res, failed_holt

#==============Holt for clusters====================
def holt_clusters(res, holt_params, loc2L, loc2S, dfl, dfs, failed_holt, test=False):
    cld_ = {}
    cls_ = {}
    locs = list(res['geohash6'].keys())
    for n, loc in enumerate(locs):
        if n % 10 == 0:
            print(f'Doing item {n}')
        cl = loc2L[loc]
        cs = loc2S[loc]
        
        l_dir = f'{HOLT_DIR}l/'
        s_dir = f'{HOLT_DIR}s/'
        ensure_dir(l_dir)
        ensure_dir(s_dir)
        
        processed_l = set([x.split('.')[0] for x in os.listdir(l_dir)])
        processed_s = set([x.split('.')[0] for x in os.listdir(s_dir)])
        tdfl = dfl[dfl['cluster']==cl]
        tdfs = dfs[dfs['cluster']==cs]
        for idx in res['geohash6'][loc]:
            if cl not in cld_:
                cld_[cl] = {}
            if cs not in cls_:
                cls_[cs] = {}
            if idx not in cld_[cl]:
                _, cl_pred_values = pred_holt(tdfl, holt_params, cl, idx, 'l', processed=processed_l, failed_holt=failed_holt, test=test)
                cld_[cl][idx] = cl_pred_values
            if idx not in cls_[cs]:
                _, cs_pred_values = pred_holt(tdfs, holt_params, cs, idx, 's', processed=processed_s, failed_holt=failed_holt, test=test)
                cls_[cs][idx] = cs_pred_values

    return cld_, cls_


# Generate holt params

def generate_params(df, failed_grp, col, type_='locs'):
    """
    Usage: 
        1.generate_params(raw_df)
        2. generate_params(dfs, train_d, failed_grp, type_='s')
        3. generate_params(dfl, type_='l')
    """
    params = ['smoothing_level', 'smoothing_slope', 'smoothing_seasonal']
    seeds = list(np.arange(4000))
    ensure_dir(f'{DIR}{type_}/')
    processed = set([x.split('.')[0] for x in os.listdir(f'{DIR}{type_}/')])
    process_holt(df, failed_grp, processed, params, seeds, type_, col)

            
def process_holt(df, failed_grp, processed, params, seeds, type_, col):
    executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
    tasks = (delayed(_process_holt)(df, failed_grp, n, loc, processed, params, seeds, type_) for n, loc in enumerate(df[col].unique()))
    res = executor(tasks)
            
def _process_holt(df, failed_grp, n, loc, processed, params, seeds, type_):
    print(f"Doing loc {n}: {loc}")
    if loc in failed_grp:
        print(f"Failed to process {loc} already in failed group")
        return
    if loc in processed:
        print(f"Skipping process {loc} in processed")
        return
    col = 'geohash6' if type_ == 'locs' else 'cluster'
    tdf = df.loc[df[col]==loc]
    tdf_p = process_grp_df(tdf)
    print(f"Length of df {loc}={len(tdf_p)}")
    if len(tdf_p.loc[tdf_p['demand'].isnull()]) > 0:
        raise ValueError
    counter = 0
    while True:
        if counter == 20:
            print(f"Failed to process {loc}")
            failed_grp.add(loc)
            return
        try:
            seed = random.choice(seeds)
            if seed+1344 >= len(tdf_p):
                raise ValueError(f'Seed value {seed} is too high')
            train = tdf_p['demand'].iloc[seed:seed+1344]
            break
        except Exception as e:
            print(e)
            counter+=1
    try:
        model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=96).fit()
        res = {p: model.params[p] for p in params}
        with open(f'{DIR}{type_}/{loc}.txt', 'w') as f:
            json.dump(res, f)
        print(f"Successfully processed loc {n}: {loc}")
    except Exception as e:
        print(f"Error Message {e}")
        print(f"Failed to parse {loc}")
        failed_grp.add(loc)




def build_params_dict_from_txt():
    types = ['locs', 'l', 's']
    res = {}
    for type_ in types:
        print(f"Doing type: {type_}")
        if type_ not in res:
            res[type_] = {}
        files = set([x.split('.')[0] for x in os.listdir(f"{DIR}{type_}/")])
        for n, grp in enumerate(files):
            if n % 10 == 0:
                print(f"Doing file {n}")
            if grp not in res[type_]:
                res[type_][grp] = {}
            
            with open(f'{DIR}{type_}/{grp}.txt', 'r') as f:
                params = json.load(f)
                res[type_][grp] = params
    return res