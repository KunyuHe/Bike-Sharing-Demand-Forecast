# Regression Pipeline: Bike Sharing Demand Forecast

This repo contains a [scikit-learn](https://scikit-learn.org/stable/) based machine learning pipeline for regression with tree-based ensemble learning algorithms, Random Forest and XGBoost in specific. It is applied on a Kaggle playground prediction competition, [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand).

The pipeline features customized preprocessing transformers that take a [pandas](https://pandas.pydata.org/) DataFrame and as input and return the encoded and scaled feature matrix and feature names. It also contains [SHAP](https://shap.readthedocs.io/en/latest/) plots that explains how and to what extent each feature contributes to the prediction to enhance the interpretability of the algorithms we employ.

<br/>

## Data

The datasets listed below can be downloaded on the competition webpage linked [here](https://www.kaggle.com/c/bike-sharing-demand/data). Users would need to create a Kaggle account and join the competition to download the datasets.

````
│  sampleSubmission.csv
│  test.csv
│  train.csv
````

The training set contains 10.9K hourly observations of bike sharing demand in Washington, D.C. on the first 20 days of each month in 2011 and 2012. It has 12 features, 9 of which are available in the test set. Refer to the data dictionary for feature definitions.

<script src="https://gist.github.com/KunyuHe/0e48ecb5ed23dce6dacfb35bcae1ae1f.js"></script>

Note that `count` is not available at prediction time as it is the target. Likewise, `casual` and `registered` are decomposition of `count` as $\text{count} = \text{casual} + \text{registered}$ and will not be available at prediction time as well.

There are 6493 observations and 10 features in the test set, accounting for hourly measurements on the last days of each month in 2011 and 2012.

Since we are working on predicting bike sharing demand in Washington, D.C., information on whether Congress and Senate were in session can be helpful. We manually scraped [Congress website](https://www.congress.gov/past-days-in-session) for past days in session data for the year 2011 and 2012 and put them in [house_sessions_1112.csv](./data/house_sessions_1112.csv) and [senate_sessions_1112.csv](./data/senate_sessions_1112.csv). We would join them with our feature matrices by datetime in the feature engineering stage. 

<br/>

## Installation

Please refer to [reg-pipe.yml](reg-pipe.yml) for package dependencies. To install the package, users would need to have [Python](https://www.python.org/downloads/) and [Anaconda](https://docs.continuum.io/anaconda/install/) installed beforehand. Users can prepare the virtual environment for the package, named `reg-pipe` by default, by running the following:

```
conda env create -f reg-pipe.yml
```

<br/>

## Usage

To run the pipeline, users would need to download the datasets as described above and unzip the file under the directory `./data`. They also need to install the dependencies as described above. Then run the following:

```
conda activate reg-pipe
cd codes
python train.py
```

<br/>

## Pipeline Overview

<br/>

## License

The package is released under the [MIT License](LICENSE).