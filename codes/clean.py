import pandas as pd
from pathlib import Path
import pickle
from utils import createLogger, createDirs

INPUT_DIR = Path('../data/')
OUTPUT_DIR = Path('../data/clean/')
LOG_DIR = Path('../logs/clean/')


def clean(file_name="train.csv"):
    mode = file_name.split(".")[0]

    df = pd.read_csv(INPUT_DIR / file_name)
    df['datetime'] = pd.to_datetime(df.datetime)
    df.set_index('datetime', drop=True, inplace=True)
    df.rename({'count': 'cnt'}, axis=1, inplace=True)

    house_sessions = pd.read_csv(INPUT_DIR / 'house_sessions_1112.csv')
    senate_sessions = pd.read_csv(INPUT_DIR / 'senate_sessions_1112.csv')
    df['house_in_session'] = pd.to_datetime(df.index.date).isin(
        pd.to_datetime(house_sessions.date)).astype(int)
    df['senate_in_session'] = pd.to_datetime(df.index.date).isin(
        pd.to_datetime(senate_sessions.date)).astype(int)

    for leakage_col in ('casual', 'registered'):
        if leakage_col in df.columns:
            df.drop(leakage_col, axis=1, inplace=True)

    cat_cols = ['season', 'holiday', 'workingday', 'weather',
                'house_in_session', 'senate_in_session']
    df[cat_cols] = df[cat_cols].astype('category')

    df.to_csv(OUTPUT_DIR / file_name, index=True)
    with open(OUTPUT_DIR / (mode + "_dtypes.pkl"), 'wb') as file:
        pickle.dump(df.dtypes.to_dict(), file)


def readClean(file_name="train.csv"):
    mode = file_name.split(".")[0]

    with open(OUTPUT_DIR / (mode + "_dtypes.pkl"), 'rb') as file:
        dtypes = pickle.load(file)

    return pd.read_csv(OUTPUT_DIR / file_name,
                       index_col=0, parse_dates=True, dtype=dtypes)


if __name__ == "__main__":
    createDirs(OUTPUT_DIR)

    createDirs(LOG_DIR)
    logger = createLogger(LOG_DIR, "clean")
    logger.info("Logger created, logging to %s" % LOG_DIR.absolute())

    clean("train.csv")
    clean("test.csv")
