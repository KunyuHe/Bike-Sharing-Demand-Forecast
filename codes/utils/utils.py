import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def createDirs(dir_path):
    """
    Create a new directory if it doesn't exist.
    Inputs:
        - dir_path: (str) path to the directory to create
    Output:
        (None)
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        if isinstance(dir_path, str):
            dir_path = Path(dir_path)
        file = open(dir_path / ".gitkeep", "w+")
        file.close()


def createLogger(log_dir, logger_name):
    """
    Create a logger and print to both console and log file.
    Inputs:
        - log_dir (str): path to the logging directory
        - logger_name (str): name of the logger
    Output:
        (Logger) a Logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(
        log_dir / (time.strftime("%Y%m%d-%H%M%S") + '.log'))
    logger.addHandler(fh)

    return logger


def saveDF(df, dir, file_name):
    df.to_csv(dir / file_name, index=True)
    with open(dir / (file_name + "_dtypes.pkl"), 'wb') as file:
        pickle.dump(df.dtypes.to_dict(), file)


def readDF(dir, file_name="train.csv"):
    with open(dir / (file_name + "_dtypes.pkl"), 'rb') as file:
        dtypes = pickle.load(file)

    return pd.read_csv(dir / file_name,
                       index_col=0, parse_dates=True, dtype=dtypes)


def ask(names, message, logger):
    """
    Ask user for their choice of an element.
    Inputs:
        - lst (list of strings): names of the element of choice
        - message (str)
    Returns:
        (int) index for either model or metrics
    """
    indices = []

    logger.info("\n" + "*" * 80 + "\nUp till now we support:")
    for i, name in enumerate(names):
        logger.info("%s. %s" % (i + 1, name))
        indices.append(str(i + 1))

    index = input("%s\n" % message)
    logger.info("*" * 80 + "\n")
    if index in indices:
        return int(index) - 1
    else:
        logger.info(
            "Input wrong. Type one in {} and hit Enter.".format(indices))
        return ask(names, message, logger)


def getNumCat(df, target):
    df = df.drop(target, axis=1)

    cat = df.columns[df.dtypes == 'category'].tolist()
    num = df.columns.difference(cat).tolist()

    return num, cat


def saveFig(dir_path, file_name, show=False):
    createDirs(dir_path)
    plt.savefig(dir_path / file_name, dpi=300, bbox_inches="tight")

    if show:
        plt.show(block=False)
        plt.pause(3)
        plt.close()


def timeStamp():
    now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    return f"Timestamp - {now}"


if __name__ == '__main__':
    pass
