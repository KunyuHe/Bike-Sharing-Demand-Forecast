import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import utils

INPUT_DIR = Path("../data/")
OUTPUT_DIR = Path("../data/clean/")
LOG_DIR = Path("../logs/clean/")


def clean(file_name="train.csv"):
    start = time.time()

    mode = file_name.split(".")[0]
    logger.info(f"({utils.timeStamp()}) Data cleaning job started on "
                f"{mode}ing data.")

    df = pd.read_csv(INPUT_DIR / file_name)
    df['datetime'] = pd.to_datetime(df.datetime)
    df.set_index('datetime', drop=True, inplace=True)

    house_sessions = pd.read_csv(INPUT_DIR / "house_sessions_1112.csv")
    senate_sessions = pd.read_csv(INPUT_DIR / "senate_sessions_1112.csv")
    df['house_in_session'] = pd.to_datetime(df.index.date).isin(
        pd.to_datetime(house_sessions.date)).astype(int)
    df['senate_in_session'] = pd.to_datetime(df.index.date).isin(
        pd.to_datetime(senate_sessions.date)).astype(int)

    if mode == "train":
        df.rename({'count': "cnt"}, axis=1, inplace=True)
        df.drop(['casual', 'registered'], axis=1, inplace=True)

    cat_cols = ['season', 'holiday', 'workingday', 'weather',
                'house_in_session', 'senate_in_session']
    df[cat_cols] = df[cat_cols].astype("category")

    utils.saveDF(df, OUTPUT_DIR, file_name)
    if mode == "train":
        logger.info("\tTarget vector histogram:")
        fig = plt.figure(figsize=(13, 6))
        plt.hist(df.cnt, color="#00245D", edgecolor="black", bins=50)
        plt.title("Training Target Distribution", fontsize=18)
        utils.saveFig(OUTPUT_DIR, "cnt_dist.png", fig, pause=5)

    logger.info(f"{mode.title()}ing feature matrix dimension: {df.shape}")
    logger.info(f"({utils.timeStamp()}) Data cleaning job on {mode}ing data"
                f" finished.")
    logger.info(f"(Time Elapsed - {(time.time() - start):.0f}s)\n\n")


def main():
    utils.createDirs(OUTPUT_DIR)
    utils.createDirs(LOG_DIR)

    global logger
    logger = utils.createLogger(LOG_DIR, "clean")
    logger.info("=" * 40 + " Data Cleaning " + "=" * 40)
    logger.info("Logger created, logging to %s" % LOG_DIR.absolute())

    clean("train.csv")
    clean("test.csv")


if __name__ == "__main__":
    main()
