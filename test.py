from preprocess import preprocess
from utils import load_pickle, save_pickle
from train import *
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import sys

def predict_cb(df, path, cols=all_features):
    #output_cols = ['geohash6'] + [f"preds_t{time}" for time in range(1,6)]
    for time in range(1,6):
        for model_id in range(5):
            logger.info(f"Predicting Catboost model t{time}_{model_id}")
            clf = CatBoostRegressor().load_model(str(path/f"cb_{time}_{model_id}"))
            current_preds = clf.predict(df[cols].values)
            df[f"cb_preds_t{time}_{model_id}"] = current_preds

def predict_dl(df, path, model_names=['final_model_v3', 'final_model_v4']):
    cat_vars = ['geohash6', 'hr', 'min', 'timestamp']
    cont_vars = [var for var in all_features if var not in cat_vars]
    logger.info("Predicting DL models")
    for n, model_name in enumerate(model_names):
        for time in range(1,6):
            logger.info(f"Predicting Catboost model t{time}_{model_name}")
            test_df = dl_preprocessing(df.copy(), time)
            learn = load_learner(path, f"{model_name}{time}.pkl",
                                test=TabularList.from_df(test_df, path=path, cat_names=cat_vars, cont_names=cont_vars))
            preds = learn.get_preds(DatasetType.Test)
            df[f'df_preds_t{time}_{n}'] = preds[0].data.numpy()

def dl_preprocessing(df, t):
    unique_cats = load_pickle('unique_cats.p')
    cat_vars = ['geohash6', 'hr', 'min', 'timestamp']
    for cat in cat_vars:
        df[cat] = df[cat].apply(lambda x: x if x in unique_cats[t][cat] else unique_cats[t][f"{cat}_mode"])
    return df

def predict_test():
    # Load files
    if len(sys.argv) < 2:
        raise RuntimeError('Please provide the filename of the testing dataset') 
    
    fn = sys.argv[1]
    strict = True if 'strict' in sys.argv else False


    #fn = 'training.csv'
    df = preprocess(fn, test=True, strict=strict)
    save_pickle(df, 'df_test.p')
    #df = load_pickle('df_test.p', 'resources/temp/')

    df.reset_index(inplace=True)
    predict_cb(df, path, cols=all_features)
    save_pickle(df, 'df_test.p')
    predict_dl(df, path)
    save_pickle(df, 'df_test.p')
    for time in range(1,6):
        df[f'preds_{time}'] = np.mean(df[[f"cb_preds_t{time}_{i}" for i in range(5)] + [f"df_preds_t{time}_{i}" for i in range(2)]], axis=1)
    save_pickle(df, 'output.p')

    df_output = df[['geohash6'] + [f'preds_{time}' for time in range(1,6)]]
    df_output.to_csv('preds.csv')
    return df

if __name__ == '__main__':
    predict_test()

