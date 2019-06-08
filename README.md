# Grab Traffic Management

This repo containes the scripts to to preprocess data as well as the trained models for test set predictions. 

## Methodology

### Model

The final model consists of 5 ensembles of 5 catboost models and 2 neural networks with categorical embeddings. The five ensembles correspond to the five time intervals T+1 to T+5, where each ensemble is used to predict the traffic demand at one time interval. The average prediction of the ensemble is the final prediction at each timestamp. 

### Feature engineering

The first feature engineering step is to allocate each geohash into 2 clusters based on geographic location - a big and small cluster `['cluster_l', 'cluster_s']`. Big clusters uses a k_means n_clusters of 50 while small clusters have an n_clusters of 150. This is to provide the model with more data about the demand in the region around each geohash, as well as provide more information for the model to learn about geohashes with very low occurrences in the dataset.

Next we add the previous 24 hours' demand from time T for every `['geohash6', 'cluster_l', 'cluster_s']`. The catboost models uses the full 24 hours demand for model training and prediction while the deep learning model uses only T-1 to T-5 demand data.

Next we add statistics for each 'location-period' pair, with locations corresponding to one of each `['geohash6', 'cluster_l', 'cluster_s']` and period being one of `['timestamp', 'day', 'hr']`. The statistics used are `['mean', 'median', 'max', 'min', 'std']`.

Lastly we perform holt winters triple exponential smoothing for each `['geohash6', 'cluster_l', 'cluster_s']`to forecast the traffic demand at each geohash-time pair for T+1 to T+5. These predictions, along with the other features above will be used as input to the ensemble.

## Usage

### Step 1: Install requirements

Navigate to the project folder in the terminal and execute: `pip install -r requirements.txt`.

### Step 2: Perform prediction

Copy the test dataset into the root directory of the project folder. The test set should contain all the data up till time T+5. The script will automatically find time T for each geohash by searching for the sixth last `geohash-timestamp` pair for each geohash. 

Note: This script assumes that every geohash in the testset contains all five T+1 to T+5 rows. If not the wrong time T will be selected because the algorithm used to identify the time T is to find the sixth latest timestamp for each geohash.

Run the prediction using the following command:

`python test.py <testset filename>`

The script will product 2 outputs in the root folder: `preds.csv` and `output.csv`. `preds.csv` contains the final predictions for each geohash, while `output.csv` will contain the final predictions as well as engineered features and each individual model's prediction.

Note: There is a strict mode that can be used with the `strict` flag, ie. `python test.py <testset filename> strict`. Without using strict, the model will automatically filter out geohashes with less than 6 occurrences in the dataset, and perform predictions with the remaining geohashes. If you use the strict flag, the script will raise a RuntimeError, informing you of the presence of these failed geohashes.




