import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import os
import matplotlib.pyplot as plt
import datetime
import joblib
from math import sin, cos, sqrt, atan2, radians
import logging
from tqdm import tqdm
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def process_features(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.drop(columns=config.DROP_COLUMNS, axis=1)
    current_year = datetime.datetime.now().year
    new_df["age"] = new_df["dob"].apply(lambda x: current_year - int(x.split("-")[0]))
    new_df = new_df.drop(columns=["dob"], axis=1)
    new_df["merchant"] = new_df["merchant"].apply(lambda x: x.split("fraud_")[1])
    new_df = new_df.rename(columns={"merchant": "merch_name"})
    new_df["full_name"] = new_df["first"] + " " + new_df["last"]
    new_df = new_df.drop(columns=["first", "last"], axis=1)
    new_df = new_df.drop(columns=["unix_time"], axis=1)
    new_df["dow"] = new_df["trans_date_trans_time"].apply(lambda x: datetime.datetime.strptime(x, config.DATETIME_FORMAT).weekday())
    new_df["dom"] = new_df["trans_date_trans_time"].apply(lambda x: int(x.split("-")[2].split(" ")[0]))
    new_df["month"] = new_df["trans_date_trans_time"].apply(lambda x: int(x.split("-")[1]))
    new_df["year"] = new_df["trans_date_trans_time"].apply(lambda x: int(x.split("-")[0]))
    def process_time(time: str) -> float:
        times = time.split(" ")[1].split(":")
        time_float = float(times[0]) + float(times[1])/60
        return time_float
    new_df["time"] = new_df["trans_date_trans_time"].apply(process_time)
    new_df = new_df.drop(columns=["trans_date_trans_time"], axis=1)
    new_df["distance"] = new_df.apply(lambda row: calculate_distance(row["lat"], row["long"], row["merch_lat"], row["merch_long"]), axis=1)
    new_df["time_interval"] = pd.cut(new_df["time"], bins=12).astype(str)
    new_df["amt_interval"] = pd.cut(new_df["amt"], bins=6).astype(str)
    new_df["age_interval"] = pd.cut(new_df["age"], bins=6).astype(str)
    new_df["city_pop_interval"] = pd.cut(new_df["city_pop"], bins=6).astype(str)
    new_df["distance_interval"] = pd.cut(new_df["distance"], bins=12).astype(str)
    new_column_order = [
        "full_name", "gender", "age", "age_interval", "job", "lat", "long", "city_pop", "city_pop_interval", "merch_name", "distance", "distance_interval", "merch_lat", "merch_long", "amt", "amt_interval", "category", "time", "time_interval", "dow", "dom", "month", "year", config.TARGET_COLUMN
    ]
    return new_df.reindex(columns=new_column_order)

def label_encode(df: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    encoder = preprocessing.LabelEncoder()
    df_copy = df.copy()
    for feat in cat_features:
        df_copy[feat] = encoder.fit_transform(df_copy[feat])
    return df_copy

def split_train_test(df: pd.DataFrame, target_class: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    for train_set, test_set in split.split(df, df[target_class]):
        strat_train_set = df.loc[train_set]
        strat_test_set = df.loc[test_set]
    return strat_train_set, strat_test_set

def select_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    copy = df.copy()
    for feat in df.columns:
        if feat not in features:
            copy = copy.drop(feat, axis=1)
    return copy

def main():
    logging.info('Loading the dataset...')
    transactions = pd.read_csv(config.DATA_PATH)
    logging.info('Processing features...')
    transactions_processed_features = process_features(transactions)
    logging.info('Encoding categorical features...')
    transactions_encoded = label_encode(transactions_processed_features, config.CAT_FEATURES)
    logging.info('Splitting train and test sets...')
    train_set, test_set = split_train_test(transactions_encoded, config.TARGET_COLUMN)
    x_train, y_train = train_set.drop(config.TARGET_COLUMN, axis=1), train_set[config.TARGET_COLUMN]
    x_train_selected = select_features(x_train, config.MODEL_FEATURES)
    logging.info('Training DecisionTreeClassifier...')
    tree = DecisionTreeClassifier(random_state=config.RANDOM_STATE)
    for _ in tqdm(range(1), desc='Fitting model', ncols=config.PROGRESS_BAR_NCOLS, colour=config.PROGRESS_BAR_COLOR):
        tree.fit(x_train_selected, y_train)
    features = list(x_train_selected.columns)
    logging.info('Saving model to %s...', config.MODEL_PATH)
    joblib.dump(value=(tree, features, config.TARGET_COLUMN), filename=config.MODEL_PATH)
    logging.info('Model trained and saved as %s', config.MODEL_PATH)

if __name__ == "__main__":
    main()
