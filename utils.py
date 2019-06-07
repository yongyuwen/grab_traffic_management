# Utilities
import logging
import datetime
import os
import pickle
import random
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format('logs', 'log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

PERIOD = 20145

def save(df, fn='resources/df_proc.f'):
    df.to_feather(fn)

def save_pickle(file, fn, dir='resources/temp/'):
    with open(dir+fn, 'wb') as f:
        pickle.dump(file, f)

def load_pickle(fn, dir='resources/pretrain/'):
    with open(dir+fn, 'rb') as f:
        res = pickle.load(f)
    return res

def get_timestamp_from_time(time):
    _, H, M = _time_to_dt(time)
    return f'{H}:{M}'

def _time_to_dt(time):
    dd = time // (24*60)
    hh = time % (24*60) // 60
    mm = time % (24*60) % 60
    return dd, hh, mm

def time_to_dt(time):
    dd, hh, mm = _time_to_dt(time)
    return datetime.datetime.strptime(f'{hh}:{mm}', '%H:%M') + datetime.timedelta(days=dd-1)

def dt_to_time(dt):
    dd, hh, mm = dt.day, dt.hour, dt.minute
    return dd*60*24 + hh*60 + mm
       
def ensure_dir(fn):
    if not os.path.exists(fn):
        os.makedirs(fn)
        
def listify(x):
    if hasattr(x, '__iter__'):
        return list(x)
    return [x]

def get_loc_cluster_d(df):
    loc2S = {}
    loc2L = {}
    S2loc = {}
    L2loc = {}
    # Get locations
    locs = list(df['geohash6'].unique())
    for n, loc in enumerate(locs):
        if n % 100 == 0:
            print(f'Doing row {n}')
        temp = df[df['geohash6']==loc].head(1).copy()
        loc2S[loc] = temp['cluster_s'].iloc[0]
        loc2L[loc] = temp['cluster_l'].iloc[0]
        #print(temp['cluster_s'].iloc[0])
        if not S2loc.get(temp['cluster_s'].iloc[0]):
            S2loc[temp['cluster_s'].iloc[0]] = []
        if not L2loc.get(temp['cluster_l'].iloc[0]):
            L2loc[temp['cluster_l'].iloc[0]] = []
            
        S2loc[temp['cluster_s'].iloc[0]].append(loc)
        L2loc[temp['cluster_l'].iloc[0]].append(loc)
        #break
    return loc2S, loc2L, S2loc, L2loc

def get_usable_indices(df, col='geohash6'):
    # Get all locations
    locs = list(df[col].unique())
    candidate_d = {}
    idx_map = {}
    failed = []
    for n, loc in enumerate(locs):
        if n % 20 == 0:
            print(f"Doing loc {n}")
        candidate_d[loc] = []
        idx_map[loc] = {n:[] for n in range(1,6)}
        temp = df[df[col]==loc]['time'].copy()
        length = len(temp)
        
        for i in range(length-5):
            #print(f'Doing row {i}')
            if temp.iloc[i]>= PERIOD:
                # Make candidate_d
                if temp.iloc[i+5] - temp.iloc[i] == 75:
                    candidate_d[loc].append(temp.iloc[i])
                
                # make idx_map
                next_five_timestamps = [temp.iloc[i+n] for n in range(1,6)]
                for j in range(1,6):
                    if temp.iloc[i]+j*15 in next_five_timestamps:
                        idx_map[loc][j].append(temp.iloc[i])
                    
            
        if len(candidate_d[loc]) == 0:
            #print(f'Location {loc} has no candidates')
            failed.append(loc)
    print('Done')
    return candidate_d, idx_map, failed


def get_train_set(loc_index_d):
    """
    Return list of indices for randomly generated training set ids
    """
    return [f"{k}-{value}" for k in loc_index_d if loc_index_d[k] is not None for value in loc_index_d[k]]

def make_training_set_d(df, idx_map, failed_grp, n=10):
    loc_index_d = {}
    # select indices 
    locs = list(df['geohash6'].unique())
    for loc in locs:
        loc_index_d[loc] = get_sample_indices(loc, idx_map, failed_grp, n=n)
    return loc_index_d

def get_sample_indices(grp, idx_map, failed_grp, n=10):
    "Randomly search for indices in candidates"
    temp = idx_map[grp]
    indices = set()
    loop_counter = 0
    while len(indices) < n:
        if loop_counter > n:
            break
        for t in temp.keys():
            if len(temp[t]) == 0:
                continue
            # Randomly select 1 item from each list
            indices.add(random.choice(temp[t]))
        loop_counter += 1
        
    if len(indices) == 0:
        print(f'No valid sets for group {grp}')
        failed_grp.add(grp)
        return None
    
    return list(indices)

def rmse_(y, pred):
    return np.sqrt(mean_squared_error(y, pred))