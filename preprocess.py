# Main preprocessing scripts
from utils import *
from holt import get_holt, holt_clusters
from train import *
import dask.dataframe as dd
import pandas as pd
import numpy as np
import datetime
import geohash2
from pyarrow import feather


def add_lat_lon(df):
    """
    Add 'latitude' and 'longitude' columns from 'geohash6'
    """
    df["lat"] = df['geohash6'].apply(lambda x: geohash2.decode_exactly(x)[0])
    df["lon"] = df['geohash6'].apply(lambda x: geohash2.decode_exactly(x)[1])
    
def proc_datetime(df):
    """
    Add 'hour', 'minute' and absolute 'time' column
    """
    df['hr'] = df['timestamp'].str.split(':').str.get(0).astype('int32')
    df['min'] = df['timestamp'].str.split(':').str.get(1).astype('int32')
    df['time'] = df['min'] + df['hr']*60 + (df['day']-1)*24*60
    
def apply_clustering(df, regions_l, regions_s):
    """ 
    Apply clustering model to assign each geohash to cluster. Clusters 
    are determined by geographic proximity
    """        
    df["cluster_s"] = regions_s.predict(df[["lat", "lon"]])
    df["cluster_l"] = regions_l.predict(df[["lat", "lon"]])
    
def add_cluster_mean(df):
    """
    Add cluster mean demand for time T 
    for large and small cluster to each row
    """
    _ld = df[['cluster_l', 'time', 'demand']].groupby(['cluster_l', 'time']).mean()
    _sd = df[['cluster_s', 'time', 'demand']].groupby(['cluster_s', 'time']).mean()
    ld = {level: _ld.xs(level).to_dict('index') for level in _ld.index.levels[0]}
    sd = {level: _sd.xs(level).to_dict('index') for level in _sd.index.levels[0]}
    df['cs_demand'] = df.apply(lambda x: sd[x['cluster_s']][x['time']]['demand'], axis=1)
    df['cl_demand'] = df.apply(lambda x: ld[x['cluster_l']][x['time']]['demand'], axis=1)
    return ld, sd

def make_cluster_df(d):
    """
    Create cluster df for holt predictions
    """
    dfl = pd.DataFrame.from_dict({(i,j): np.mean(d[i][j]['demand']) 
                                  for i in d.keys() 
                                  for j in d[i].keys()},
                                 orient='index')
    dfl = dfl.reset_index()
    dfl['cluster'] = dfl['index'].apply(lambda x: x[0])
    dfl['time'] = dfl['index'].apply(lambda x: x[1])
    dfl["timestamp"] = dfl["time"].apply(lambda x: get_timestamp_from_time(x))
    dfl.rename(columns={0:'demand'}, inplace=True)
    dfl.drop('index', inplace=True, axis=1)
    dfl = dfl.sort_values(['cluster', 'time'])
    return dfl


def add_demand(df_train, ds, prefixes=['t', 'cl', 'cs'], cols=['geohash6', 'cluster_l', 'cluster_s']):
    """
    Add demand for time t-96 (previous day) to t+0 for each geohash, large cluster and small cluster
    """
    for delta in range(-96, 1):
        for prefix, col, d in zip(prefixes, cols, ds):
            logger.info(f"Adding demand for {prefix}:{delta}")
            df_train[f'{prefix}{delta}'] = df_train.apply(lambda x: _add_demand(x[col], x['time'], delta, prefix, col, d), axis=1)

def _add_demand(grp, time, delta, prefix, col, d):
    """
    Helper function for add_demand function
    """
    if prefix == 't':
        if f"{grp}-{int(time)+int(delta)*15}" in d.index:
            return d.loc[f"{grp}-{int(time)+int(delta)*15}"]['demand']
        return np.nan
    elif prefix == 'cl' or prefix == 'cs':
        return d[grp][time]['demand']
    else:
        return NameError('An error occured when adding demand')

def get_last_time(df):
    """
    Find the last timestamp for each geohash
    """
    temp = df[['geohash6', 'time']].groupby('geohash6').max()
    temp_2 = temp.reset_index()
    temp_2 = temp_2.set_index(temp_2.apply(lambda x: f"{x['geohash6']}-{x['time']}", axis=1)).index
    return temp.to_dict()['time'], list(temp_2)

def get_time_t(df, strict=True):
    """
    Find the row in the df corresponding to time = T for each geohash
    If there are geohashes that have less than 5 occurrences, exclude them from the resultant df, 
    return them and raise a warning 
    """
    temp = df[['geohash6', 'time']].groupby('geohash6').agg(find_item_from_idx)
    res = temp.loc[temp['time'].notna()].astype('int')
    nas = temp.loc[temp['time'].isna()]
    if len(nas) > 0 and strict:
        raise RuntimeError(f"""There are {len(nas)} geohashes without the full T+1 to T+5 rows. Please resolve
            this manually. To automatically remove these rows and run predictions on the remaining geohashes,
            use the command line argument 'strict' """)
    res_idx = res.reset_index()
    res_idx = res_idx.set_index(res_idx.apply(lambda x: f"{x['geohash6']}-{x['time']}", axis=1)).index
    return res.to_dict()['time'], list(res_idx), nas


def find_item_from_idx(series):
    try:
        return series.sort_values().iloc[-6]
    except:
        return None
                                                                                   
def add_statistics(df_train, raw_df, test=False, agg_ds=None):
    """
    Feature engineering to add location-time statistics.
    If not test (ie. train): build statistics dictionary
    If test: apply stats from trainset
    """
    locs = ['geohash6', 'cluster_l', 'cluster_s']
    periods = ['timestamp', 'day', 'hr']
    agg_fns = ['mean', 'median', 'max', 'min', 'std']
    agg_ds_ = {} if not test else agg_ds.copy()
    for loc in locs:
        for period in periods:
            for agg in agg_fns:
                logger.info(f"Doing loc: {loc} period: {period} agg: {agg}")
                process_agg(df_train, raw_df, loc, period, agg, agg_ds_, test)
    return agg_ds_
                                           
        
def process_agg(df_train, raw_df, loc, period, agg, agg_ds_, test=False):
    """
    Helper function for adding statistics
    """
    if not test:
        agg_df = raw_df[[loc, period, 'demand']].groupby([loc, period]).agg(agg)
        agg_d = {level: agg_df.xs(level).to_dict('index') for level in agg_df.index.levels[0]}
        df_train[f'{loc}_{period}_{agg}'] = df_train.apply(lambda x: agg_d[x[loc]][x[period]]['demand'], axis=1)
        agg_ds_[f'{loc}_{period}_{agg}'] = agg_d
        return 
    if not agg_ds_:
        raise ValueError('Need to have trainset statistics if building test set')
    if period == 'day':
        agg_df = raw_df[[loc, period, 'demand']].groupby([loc, period]).agg(agg)
        agg_d = {level: agg_df.xs(level).to_dict('index') for level in agg_df.index.levels[0]}
    else:
        agg_d = agg_ds_[f'{loc}_{period}_{agg}']
    df_train[f'{loc}_{period}_{agg}'] = df_train.apply(lambda x: agg_d[x[loc]][x[period]]['demand'], axis=1)

                                           
def _append_holt(row, holt_res, cls_, cld_, type_, i):
    if type_ == 'geohash6':
        val = holt_res['geohash6'][row['geohash6']][row['time']]
    elif type_ == 'cld':
        try:
            val = cld_[row['cluster_l']][row['time']]
        except:
            logger.info(row['geohash6'])
            logger.info(row['cluster_l'])
            logger.info(row['time'])
            raise NameError
    elif type_ == 'cls':
        val = cls_[row['cluster_s']][row['time']]
    if isinstance(val, np.ndarray): 
        return val[i+5]
    logger.warning(f"Cannot find predictions for {row['geohash6']} at time {row['time']}")
    return None


def append_holt(df, holt_res, cls_, cld_):
    for i in range(-5, 6):
        logger.info(f"Doing time interval {i}")
        df[f'holt_t_{i}'] = df.apply(lambda row: _append_holt(row, holt_res, cls_, cld_, 'geohash6', i), axis=1)
        df[f'holt_l_{i}'] = df.apply(lambda row: _append_holt(row, holt_res, cls_, cld_, 'cld', i), axis=1)
        df[f'holt_s_{i}'] = df.apply(lambda row: _append_holt(row, holt_res, cls_, cld_, 'cls', i), axis=1)

    
def preprocess(fn, test=False, strict=True):
    suffix = '_test' if test else '_train'

    logger.info("Loading required files")
    raw_df = pd.read_csv(fn) 
    holt_params = load_pickle('holt_params.p')
    agg_ds = load_pickle('agg_ds.p') if test else None
    regions_l = load_pickle('regions_l.p')
    regions_s = load_pickle('regions_s.p')
    failed_grp = load_pickle('failed_grp.p')


    # # Basic feature engineering
    # logger.info("Adding lat lon")
    # add_lat_lon(raw_df)

    # logger.info("Adding datetime")
    # proc_datetime(raw_df)

    # logger.info("Adding clusters")
    # apply_clustering(raw_df, regions_l, regions_s)

    # logger.info("Adding cluste rmean")
    # ld, sd = add_cluster_mean(raw_df)
    # save_pickle(ld, 'ld.p')
    # save_pickle(sd, 'sd.p')

    # logger.info("Making dfl and dfs")
    # dfl = make_cluster_df(ld)
    # dfs = make_cluster_df(sd)
    # save_pickle(dfl, 'dfl.p')
    # save_pickle(dfs, 'dfs.p')

    # logger.info("Create location_cluster mapping")
    # loc2S, loc2L, S2loc, L2loc = get_loc_cluster_d(raw_df)
    # save_pickle(loc2S, 'loc2S.p')
    # save_pickle(loc2L, 'loc2L.p')
    # save_pickle(S2loc, 'S2loc.p')
    # save_pickle(L2loc, 'L2loc.p')


    # logger.info("Reset indices")
    # raw_df = raw_df.set_index(raw_df.apply(lambda x: f"{x['geohash6']}-{x['time']}", axis=1))
    # save_pickle(raw_df, 'raw_df.p')
    raw_df = load_pickle('raw_df.p', 'resources/temp/')
    ld = load_pickle('ld.p', 'resources/temp/')
    sd = load_pickle('sd.p', 'resources/temp/')
    loc2L = load_pickle('loc2L.p', 'resources/temp/')
    loc2S = load_pickle('loc2S.p', 'resources/temp/')
    dfl = load_pickle('dfl.p', 'resources/temp/')
    dfs = load_pickle('dfs.p', 'resources/temp/')

    # Select actual set
    if test:
        logger.info("Making testset indices")
        loc_index_d, usable_indices, nas = get_time_t(raw_df, strict=strict)
        save_pickle(loc_index_d, 'loc_index_d.p')
        save_pickle(usable_indices, 'usable_indices.p')
    else:
        logger.info("Making trainset indices")
        raw_df.sort_values(['geohash6', 'time'], inplace=True)
        candidate_d, idx_map, failed = get_usable_indices(raw_df)
        #idx_map = load_pickle('idx_map.p')


        logger.info("Make Training set")
        loc_index_d = make_training_set_d(raw_df, idx_map, failed_grp, n=1000)
        save_pickle(loc_index_d, 'loc_index_d.p')

        logger.info("Make usable indices")
        usable_indices = get_train_set(loc_index_d)
        save_pickle(usable_indices, 'usable_indices.p')
                                           
    df = raw_df.loc[usable_indices]
    logger.info(f"Length of df = {len(df)}")
    save_pickle(df, f'df{suffix}.p')

    logger.info("Adding demand")                                   
    add_demand(df, [raw_df, ld, sd])
    save_pickle(df, f'df{suffix}.p')

    logger.info("Adding statistics")  
    add_statistics(df, raw_df, test=test, agg_ds=agg_ds)
    save_pickle(df, f'df{suffix}.p')
    '''
    raw_df = load_pickle('raw_df_test.p')
    dfl = load_pickle('dfl_test.p')
    dfs = load_pickle('dfs_test.p')
    ld = load_pickle('ld_test.p')
    sd = load_pickle('sd_test.p')
    loc2L = load_pickle('loc2L_test.p')
    loc2S = load_pickle('loc2S_test.p')
    df = load_pickle('df_test.p')
    loc_index_d = load_pickle('loc_index_d_test.p')
  
    df_train = load_pickle('df_train.p')
    raw_df = load_pickle('raw_df.p')
    loc_index_d = load_pickle('loc_index_d.p')
    dfl = load_pickle('dfl.p')
    dfs = load_pickle('dfs.p')
    loc2L = load_pickle('loc2L.p')
    loc2S = load_pickle('loc2S.p')
    res = load_pickle('res.p')
    failed_holt = load_pickle('failed_holt.p')
    '''

    # Add holt features
    logger.info("Generating holt")
    res, failed_holt = get_holt(raw_df, holt_params, loc_index_d, failed_grp, test=test)
    save_pickle(res, 'holt_res.p')
    save_pickle(failed_holt, 'failed_holt.p')

    logger.info("Generating cluster holt")
    cld_, cls_ = holt_clusters(res, holt_params, loc2L, loc2S, dfl, dfs, failed_holt=failed_holt, test=test)
    save_pickle(cld_, 'cld_.p')
    save_pickle(cls_, 'cls_.p')   
    
    logger.info("Appending holt")
    append_holt(df, res, cls_, cld_)
    save_pickle(df, f'df{suffix}.p')

    return df

    

    


