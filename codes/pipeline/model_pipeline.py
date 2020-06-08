import sys
import time
from datetime import datetime

import joblib
import numpy as np
import xgboost
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import shap

sys.path.append("../")
from utils import utils, evaluate
from pipeline.preprocess_pipeline import getPreprocessingSteps
from codes.train import INPUT_DIR, OUTPUT_DIR


class ModelPipeline():
    LOGGER = None
    MODEL_NAMES = ["Dummy", "Random Forest", "XgBoost"]
    MODELS = [DummyRegressor, RandomForestRegressor, xgboost.XGBRegressor]
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
        self.preprocessor = getPreprocessingSteps(num, cat)

        # Feature matrix and target
        self.y_train = train[self.target].values
        self.X_train = train.drop(self.target, axis=1)

        self.X_test = utils.readDF(INPUT_DIR, "test.csv")

    def construct(self):
        self.pip = Pipeline(steps=[
            ('pre', self.preprocessor),
            ('reg', self.model(**self.params['default']))
        ])

        return self

    def tune(self, k):
        hyper = self.params['hyper']
        search_space = [{'reg__' + arg: val for arg, val in dct.items()}
                        for dct in hyper]

        grid = GridSearchCV(
            self.pip,
            param_grid=search_space,
            cv=k,
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
            verbose=1,
            n_jobs=-1
        )

        logger, now, base = self.get_logger()
        logger.info(base + "Hyperparameter tuning job initiated.")
        logger.info(f"\tModel parameters: {self.params['default']};")
        logger.info(f"\tSearching space: {self.params['hyper']}")

        start = time.time()
        logger.info("*" * 80)
        grid.fit(self.X_train, self.y_train)

        self.est = grid.best_estimator_
        self.params = grid.best_params_
        self.cv_score = grid.best_score_

        end = time.time()
        logger.info("*" * 80 +
                    f"\n\tBest set of parameters: {self.params}.")
        logger.info(
            f"\tCross-validation RMSLE: {-1 * self.cv_score:.3f}.")
        logger.info(base + ("Tuning job finished. "
                            f"Time elapsed {end - start:.2f}s."))

        self.model_dir = OUTPUT_DIR / (self.model_name.lower() + "/" + now)
        utils.createDirs(self.model_dir)
        joblib.dump(self.est['reg'], self.model_dir / "model.joblib",
                    compress=1)
        logger.info(f"\nModel saved to {self.model_dir.absolute()}.")

    def evaluate(self):
        logger, now, base = self.get_logger()
        logger.info("\n" + base + "Model evaluation job initiated.")
        start = time.time()

        feature_names = np.array([name.split("__")[1]
                         for name in self.est['pre'].get_feature_names()])
        pre = self.est['pre']
        reg = self.est['reg']

        logger.info("\tMaking predictions on the test set.")
        ypred = np.exp(self.est.predict(self.X_train))
        np.savetxt(self.model_dir / "pred.csv", ypred, delimiter=",")
        evaluate.plotPredictedValues(np.exp(self.y_train), ypred,
                                     self.model_dir)
        logger.info(f"\tPredictions saved under {self.model_dir}.")

        if hasattr(reg, "feature_importances_"):
            logger.info("\tPlotting top 10 important features:")
            importances = reg.feature_importances_
            evaluate.plotFeatureImportances(importances, feature_names,
                                            self.model_dir)

            logger.info("\tApplying SHAP for model interpretability:")
            logger.info("*" * 80)
            sample = shap.sample(self.X_test, 300, random_state=123)
            sample = pre.transform(sample).toarray()
            evaluate.treeExplain(reg, sample, feature_names, self.model_dir)
            logger.info("*" * 80)
        logger.info(base + ("Model evaluation job finished. "
                            f"Time elapsed {time.time() - start:.2f}s."))

    def get_logger(self):
        logger = ModelPipeline.LOGGER
        now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        base = f"(Timestamp - {now}, Model - {self.model_name}) "

        return logger, now, base

    @classmethod
    def add_model(cls, model, model_name):
        ModelPipeline.MODELS.append(model)
        ModelPipeline.MODEL_NAMES.append(model_name)

    @classmethod
    def configure_hyper(cls, params):
        ModelPipeline.PARAMS = params

    @classmethod
    def configure_logger(cls, logger):
        ModelPipeline.LOGGER = logger

    @classmethod
    def get_model_names(cls):
        return cls.MODEL_NAMES
