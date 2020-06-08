import sys
import time
from pathlib import Path

import numpy as np

sys.path.append("../")
from utils import utils
from train import INPUT_DIR

OUTPUT_DIR = INPUT_DIR
INPUT_DIR = Path('../data/clean/')
LOG_DIR = Path('../logs/feature-engineer/')


def featureEngineer(file_name="train.csv"):
    start = time.time()

    mode = file_name.split(".")[0]
    logger.info(f"({utils.timeStamp()}) Feature engineering job started on "
                f"{mode}ing data.")

    df = utils.readDF(INPUT_DIR, file_name)

    # Transform target to account for RMSLE
    if mode == "train":
        df.cnt = np.log(df.cnt + 1)

    # Add time-related features
    df['year'] = df.index.year.astype('category')
    df['month'] = df.index.month.astype('category')
    df['dayofweek'] = df.index.dayofweek.astype('category')
    df['hour'] = df.index.hour.astype('category')

    # pairs = ((('dayofweek', 'hour'), 'category'),
    #          (('holiday', 'month'), 'category'),
    #          (('temp', 'humidity'), np.float64),
    #          (('temp', 'windspeed'), np.float64),
    #          (('temp', 'hour'), np.float64))
    # for pair, dtype in pairs:
    #     cross = dmatrix('{}*{}-1'.format(*pair), df,
    #                     return_type="dataframe").astype(dtype)
    #     interactions = cross[[col for col in cross.columns if ":" in col]]
    #     df = pd.concat((df, interactions), axis=1)

    df['temp_diff'] = df.atemp - df.temp

    # Drop redundant features
    df.drop(['season', 'workingday', 'atemp'], axis=1, inplace=True)
    utils.saveDF(df, OUTPUT_DIR, file_name)

    logger.info(f"\t{mode.title()}ing feature matrix dimension: {df.shape}")
    logger.info(f"({utils.timeStamp()}) Feature engineering job on {mode}ing "
                f"data finished. Time elapsed {(time.time() - start):.2f}s.\n")


def main():
    utils.createDirs(OUTPUT_DIR)
    utils.createDirs(LOG_DIR)

    global logger
    logger = utils.createLogger(LOG_DIR, "feature-engineer")
    logger.info("\n" + "=" * 40 + " Feature Engineering " + "=" * 40)
    logger.info("Logger created, logging to %s" % LOG_DIR.absolute())

    featureEngineer(file_name="train.csv")
    featureEngineer(file_name="test.csv")
