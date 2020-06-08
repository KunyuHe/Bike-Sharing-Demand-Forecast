import argparse
import json
import warnings
from pathlib import Path

from utils import utils

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--version', dest='version', type=int, default=1,
                    help=("Model version. Update when the feature "
                          "engineering or preprocessing steps change."))
parser.add_argument('--k', dest='k', type=int, default=5,
                    help="Specify k for k-fold cross-validation.")
parser.add_argument('--config_file', dest='config_file', type=str,
                    default='kunyu_config.json',
                    help=("Specify the name of the configuration file for "
                          "hyperparameter tuning."))
args = parser.parse_args()

INPUT_DIR = Path(f'../data/feature-engineer/version-{args.version}/')
CONFIG_DIR = Path('../configs/')
OUTPUT_DIR = Path(f'../models/version-{args.version}/')
LOG_DIR = Path('../logs/train/')

if __name__ == '__main__':
    # Import here to avoid circular imports
    from prepare import clean, feature_engineering
    from pipeline.model_pipeline import ModelPipeline

    clean.main()
    feature_engineering.main()

    utils.createDirs(OUTPUT_DIR)
    utils.createDirs(LOG_DIR)

    logger = utils.createLogger(LOG_DIR, "model")
    logger.info("\n" + "=" * 40 + " Model Training " + "=" * 40)
    logger.info("Logger created, logging to %s" % LOG_DIR.absolute())

    # Configure the modeling pipeline
    ModelPipeline.configure_logger(logger)
    with open(CONFIG_DIR / args.config_file, 'r') as file:
        params = json.load(file)
        ModelPipeline.configure_hyper(params)

    model_index = utils.ask(ModelPipeline.get_model_names(),
                            "Please specify the training model by index:",
                            logger)
    regressor = ModelPipeline(model_index=model_index).construct()
    regressor.tune(args.k)
    regressor.evaluate()
