from pathlib import Path

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (StandardScaler, LabelBinarizer,
                                   PolynomialFeatures)

from utils import createLogger, createDirs, FeatureNames

INPUT_DIR = Path('../data/preprocessed/')
OUTPUT_DIR = Path('../data/preprocessed/')
LOG_DIR = Path('../logs/preprocessed/')


class CategoricalEncoder(TransformerMixin):
    def __init__(self):
        self.binary = None
        self.multinomial = None
        self.encs = {}

    def fit(self, X, y=None):
        self.binary = X.columns[X.nunique() == 2].tolist()
        self.multinomial = X.columns.difference(self.binary).tolist()

        for column in X.columns:
            self.encs[column] = LabelBinarizer().fit(X[column])

        return self

    def transform(self, X):
        encoded, colnames = [], []

        for column in X.columns:
            encoded.append(self.encs[column].transform(X[column]))
            levels = self.encs[column].classes_

            if len(levels) == 2:
                colnames.append(column)
            else:
                colnames += ["%s_%s" % (column, level) for level in levels]

        FeatureNames(OUTPUT_DIR).append(colnames)
        return np.concatenate(encoded, axis=1)


def get_preprocessing_steps(numerical_lst, categorical_lst):
    preprocessing = FeatureUnion([
        ('numerical_features',
         ColumnTransformer([('numericals',
                             Pipeline(steps=[
                                 ('scale', StandardScaler()),
                                 ('poly', PolynomialFeatures())
                             ]),
                             numerical_lst)])
         ),
        ('categorical_features',
         ColumnTransformer([('categoricals',
                             Pipeline(steps=[
                                 ('encode', CategoricalEncoder())
                             ]),
                             categorical_lst)])
         )
    ])

    return preprocessing


if __name__ == '__main__':
    createDirs(OUTPUT_DIR)
    createDirs(LOG_DIR)

    logger = createLogger(LOG_DIR, "preprocessing")
    logger.info("=" * 30 + "Data Preprocessing" + "=" * 30)
    logger.info("Logger created, logging to %s" % LOG_DIR.absolute())

    logger.info(
        "Feature names saved to %s\\feature_names.txt" % LOG_DIR.absolute())
    logger.info("Preprocessing Pipeline:")
    logger.info(get_preprocessing_steps([], []))
