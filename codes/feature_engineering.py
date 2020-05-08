from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix

from utils import createLogger, createDirs, saveDF, readDF

INPUT_DIR = Path('../data/clean/')
OUTPUT_DIR = Path('../data/preprocessed/')
LOG_DIR = Path('../logs/preprocessed/')


def featureEngineer(file_name="train.csv"):
    df = readDF(INPUT_DIR, file_name)

    # Add time-related features
    df['year'] = df.index.year.astype('category')
    df['month'] = df.index.month.astype('category')
    df['dayofweek'] = df.index.dayofweek.astype('category')
    df['hour'] = df.index.hour.astype('category')

    pairs = ((('dayofweek', 'hour'), 'category'),
             (('holiday', 'month'), 'category'),
             (('temp', 'humidity'), np.float64),
             (('temp', 'windspeed'), np.float64),
             (('temp', 'hour'), np.float64))
    for pair, dtype in pairs:
        cross = dmatrix('{}*{}-1'.format(*pair), df,
                        return_type="dataframe").astype(dtype)
        interactions = cross[[col for col in cross.columns if ":" in col]]
        df = pd.concat((df, interactions), axis=1)

    df['temp_diff'] = df.atemp - df.temp

    # Drop redundant features
    df.drop(['season', 'workingday', 'atemp'], axis=1, inplace=True)
    saveDF(df, OUTPUT_DIR, file_name)


if __name__ == '__main__':
    createDirs(OUTPUT_DIR)
    createDirs(LOG_DIR)

    logger = createLogger(LOG_DIR, "feature_engineering")
    logger.info("=" * 40 + "Feature Engineering" + "=" * 40)
    logger.info("Logger created, logging to %s" % LOG_DIR.absolute())

    featureEngineer(file_name="train.csv")
    featureEngineer(file_name="test.csv")
