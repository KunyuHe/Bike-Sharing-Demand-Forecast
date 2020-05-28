import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import xgboost
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

warnings.simplefilter(action='ignore', category=FutureWarning)

from preprocess_pipeline import get_preprocessing_steps
import utils
import clean
import feature_engineering

INPUT_DIR = Path('../data/preprocessed/')
OUTPUT_DIR = Path('../models/')
LOG_DIR = Path('../logs/model/')
CONFIG_DIR = Path('../configs/')


class ModelPipeline():
    MODEL_NAMES = ["Dummy", "Regularized Linear Regression",
                   "Support Vector Machine", "Random Forest", "XgBoost"]
    MODELS = [DummyRegressor, SGDRegressor, SVR, RandomForestRegressor,
              xgboost.XGBRegressor]
    PARAMS = {name: dict() for name in MODEL_NAMES}

    def __init__(self, target='cnt', model_index=0):
        # Model and parameters
        self.target = target
        self.model_index = model_index
        self.model = ModelPipeline.MODELS[self.model_index]
        self.model_name = ModelPipeline.MODEL_NAMES[self.model_index]
        try:
            self.params = ModelPipeline.PARAMS[self.model_name]
        except:
            raise KeyError(f"Configuration file does not host parameters for"
                           f"{self.model_name}")

        # Feature names and preprocessing pipeline
        train = utils.readDF(INPUT_DIR, "train.csv")
        num, cat = utils.getNumCat(train, self.target)
        self.preprocessing = get_preprocessing_steps(num, cat)
        with open(INPUT_DIR / "numerical_features.txt", 'w') as file:
            file.write("\n".join(num))

        # Feature matrix and target
        self.y_train = train[self.target].values
        self.X_train = train.drop(self.target, axis=1)

        self.X_test = utils.readDF(INPUT_DIR, "test.csv")

    def construct(self):
        self.pip = Pipeline(steps=[
            ("preprocess", self.preprocessing),
            ('reg', self.model(**self.params['default']))
        ])

        return self

    def tune(self, k):
        hyper = self.params['hyper']
        search_space = [{'reg__' + arg: val for arg, val in dct.items()}
                        for dct in hyper]

        self.grid = GridSearchCV(
            self.pip,
            param_grid=search_space,
            cv=k,
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
            verbose=0
        )

        now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        base = f"(Timestamp - {now}, Model - {self.model_name})"
        logger.info(base + " Hyperparameter tuning job initiated.")
        logger.info(f"\tModel parameters: {self.params['default']};")
        logger.info(f"\tSearching space: {self.params['hyper']}")

        start = time.time()
        logger.info("\tTuning job started.")
        self.grid.fit(self.X_train, self.y_train)

        end = time.time()
        logger.info(f"\tBest set of parameters: {self.grid.best_params_}.")
        logger.info(
            f"\tCross-validation RMSLE: {-1 * self.grid.best_score_:.3f}.")
        logger.info(
            base + " Tuning job finished. Elapsed time {:.3f} s.".format(
                int(end - start)))

        utils.createDirs(self.model_name.lower())
        model_dir = OUTPUT_DIR / (self.model_name.lower() + "/" + now)
        utils.createDirs(model_dir)
        joblib.dump(self.grid.best_estimator_['reg'],
                    model_dir / "model.joblib",
                    compress=1)
        logger.info(base + f" Model saved to {model_dir.absolute()}.\n\n")

    @classmethod
    def add_model(cls, model, model_name):
        ModelPipeline.MODELS.append(model)
        ModelPipeline.MODEL_NAMES.append(model_name)

    @classmethod
    def configure(cls, params):
        ModelPipeline.PARAMS = params


if __name__ == '__main__':
    clean.main()
    feature_engineering.main()

    utils.createDirs(OUTPUT_DIR)
    utils.createDirs(LOG_DIR)

    logger = utils.createLogger(LOG_DIR, "model")
    logger.info("=" * 40 + " Modeling " + "=" * 40)
    logger.info("Logger created, logging to %s" % LOG_DIR.absolute())

    parser = argparse.ArgumentParser()
    parser.add_argument('--ask', dest='ask_user', type=int, default=1,
                        help=(
                            "Whether the script should ask for user input to "
                            "run cross-validation with a specific regressor or "
                            "all of them"))
    parser.add_argument('--k', dest='k', type=int, default=5,
                        help="Specify k for k-fold cross-validation.")
    parser.add_argument('--config_file', dest='config_file', type=str,
                        default='kunyu_config.json',
                        help=("Specify the name of the configuration file for "
                              "hyperparameter tuning."))
    args = parser.parse_args()
    args.ask_user = bool(args.ask_user)

    with open(CONFIG_DIR / args.config_file, 'r') as file:
        params = json.load(file)
        ModelPipeline.configure(params)

    if args.ask_user:
        model_index = utils.ask(ModelPipeline.MODEL_NAMES,
                                "Please specify the training model by index:",
                                logger)
        regressor = ModelPipeline(model_index=model_index).construct()
        regressor.tune(args.k)
    else:
        pass
