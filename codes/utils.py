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


def readFeatureNames(dir):
    with open(dir / "feature_names.txt", 'r') as file:
        lst = file.readlines()
    return lst


class FeatureNames():
    def __init__(self, dir):
        self.dir = dir

    def write(self, feature_names):
        with open(self.dir / "feature_names.txt", 'w') as file:
            file.write("\n".join(feature_names))

    def append(self, feature_names):
        with open(self.dir / "feature_names.txt", 'a') as file:
            file.write("\n")
            file.write("\n".join(feature_names))

    def read(self):
        with open(self.dir / "feature_names.txt", 'w') as file:
            lst = file.readlines()
            print(lst)


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

    logger.info("Up till now we support:")
    for i, name in enumerate(names):
        logger.info("%s. %s" % (i + 1, name))
        indices.append(str(i + 1))

    index = input("%s\n" % message)
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


def saveFig(dir_path, file_name, fig, pause=3):
    createDirs(dir_path)
    fig.savefig(dir_path / file_name, dpi=300, bbox_inches="tight")

    fig.tight_layout()
    plt.show(block=False)
    plt.pause(pause)
    plt.close()


def timeStamp():
    now = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    return f"Timestamp - {now}"


if __name__ == '__main__':
    pass
