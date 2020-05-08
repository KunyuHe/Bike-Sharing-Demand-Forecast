import argparse
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from preprocess_pipeline import get_preprocessing_steps
from utils import (readDF, ask, getNumCat, FeatureNames, createDirs,
                   createLogger)

INPUT_DIR = Path('../data/preprocessed/')
OUTPUT_DIR = Path('../data/model/')
LOG_DIR = Path('../logs/model/')


class ModelPipeline():
    MODEL_NAMES = ["MeanDummy", "Random Forest Regressor"]
    MODELS = [DummyRegressor, RandomForestRegressor]

    def __init__(self, target='cnt', model_index=0):
        self.target = target
        self.model_index = model_index

        train = readDF(INPUT_DIR, "train.csv")
        self.numericals, self.categoricals = getNumCat(train, self.target)
        FeatureNames(INPUT_DIR).write(self.numericals)

        self.y_train = train[self.target].values
        self.X_train = train.drop(self.target, axis=1)

        self.X_test = readDF(INPUT_DIR, "test.csv")

    def construct(self):
        preprocessing = get_preprocessing_steps(self.numericals,
                                                self.categoricals)
        self.pip = Pipeline(steps=[
            ("preprocess", preprocessing),
            ('reg', self.MODELS[self.model_index]())
        ])

        return self

    def crossVal(self, k):
        cv_score = cross_val_score(self.pip,
                                   self.X_train, self.y_train,
                                   scoring=make_scorer(mean_squared_log_error),
                                   cv=k)
        return np.mean(cv_score)


if __name__ == '__main__':
    createDirs(OUTPUT_DIR)
    createDirs(LOG_DIR)

    logger = createLogger(LOG_DIR, "feature_engineering")
    logger.info("=" * 40 + "Feature Engineering" + "=" * 40)
    logger.info("Logger created, logging to %s" % LOG_DIR.absolute())

    parser = argparse.ArgumentParser()
    parser.add_argument('--ask', dest='ask_user', type=int, default=1,
                        help=(
                            "Whether the script should ask for user input to "
                            "run cross-validation with a specific regressor or "
                            "all of them"))
    parser.add_argument('--k', dest='k', type=int, default=3,
                        help="Specify k for k-fold cross-validation.")
    parser.add_argument('--config_file', dest='config_file', type=str,
                        default='kunyu_config.json',
                        help=("Specify the name of the configuration file for ",
                              "hyperparameter tuning."))
    args = parser.parse_args()
    args.ask_user = bool(args.ask_user)

    if args.ask_user:
        model_index = ask(ModelPipeline.MODEL_NAMES,
                          "Please specify the training model by index:",
                          logger)
        regressor = ModelPipeline(model_index=model_index).construct()
        logger.info("Cross-validation MSLE: %.4f" % regressor.crossVal(args.k))
    else:
        pass
