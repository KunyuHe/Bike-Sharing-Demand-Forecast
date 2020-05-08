from pathlib import Path

import pandas as pd

from utils import createLogger, createDirs, saveDF

INPUT_DIR = Path('../data/')
OUTPUT_DIR = Path('../data/clean/')
LOG_DIR = Path('../logs/clean/')


def clean(file_name="train.csv"):
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

    if "train" in file_name:
        df.drop(['casual', 'registered'], axis=1, inplace=True)

    cat_cols = ['season', 'holiday', 'workingday', 'weather',
                'house_in_session', 'senate_in_session']
    df[cat_cols] = df[cat_cols].astype('category')

    saveDF(df, OUTPUT_DIR, file_name)


if __name__ == "__main__":
    createDirs(OUTPUT_DIR)

    createDirs(LOG_DIR)
    logger = createLogger(LOG_DIR, "clean")
    logger.info("=" * 40 + "Data Cleaning" + "=" * 40)
    logger.info("Logger created, logging to %s" % LOG_DIR.absolute())

    clean("train.csv")
    clean("test.csv")
