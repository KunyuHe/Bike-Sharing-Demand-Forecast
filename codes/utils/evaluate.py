import sys

import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.font_manager import FontProperties

sys.path.append("../")
from utils import utils

LASALLE = "#00245D"
GOLD = "#A48D68"

TITLE = FontProperties(family="Arial", size=18, weight="semibold")
AXIS = FontProperties(family="Arial", size=14)
TICKS = FontProperties(family="Arial", size=12)


def plotPredictedValues(ytrue, ypred, dir_path, title=""):
    fig = plt.figure(figsize=(13, 6))

    plt.hist(ytrue, bins=50, color=LASALLE, edgecolor="black", label="True y")
    plt.hist(ypred, bins=50, alpha=0.8, color=GOLD, edgecolor="black",
             label="Predicted y")

    plt.legend(prop=TICKS)
    plt.xlabel("True/Predicted Bike Sharing Demand", fontproperties=AXIS)
    plt.ylabel("Count", fontproperties=AXIS)
    plt.title(title, fontproperties=TITLE)

    utils.saveFig(dir_path, "true_pred.png", show=True)


def plotFeatureImportances(importances, feature_names, dir_path, top_n=10):
    """
    Plot the feature importance of the classifier if it has this attribute. This
    credit to the University of Michigan.
    Inputs:
        - importances (array of floats): feature importances
        - col_names (list of strings): feature names
        - dir_path (str): path of the directory for training visualization
        - top_n (int): number of features with the highest importances to keep
        - title (string): the name of the model
    """
    indices = np.argsort(importances)[::-1][:top_n]
    labels = feature_names[indices][::-1]

    plt.figure(figsize=(13, 6))
    plt.barh(range(top_n), sorted(importances, reverse=True)[:top_n][::-1],
             color='g', alpha=0.4, edgecolor=['black'] * top_n)

    plt.xlabel("Feature Importance", fontproperties=AXIS)
    plt.ylabel("Feature Name", fontproperties=AXIS)
    plt.yticks(np.arange(top_n), labels, fontproperties=AXIS)

    utils.saveFig(dir_path, "feature importance.png", show=True)


def treeExplain(reg, sample, feature_names, dir_path):
    explainer = shap.TreeExplainer(reg, sample)
    shap_values = explainer.shap_values(sample)

    shap.force_plot(explainer.expected_value, shap_values[0, :],
                    sample[0, :], feature_names=feature_names,
                    show=False, matplotlib=True)
    utils.saveFig(dir_path, "force_plot.png", show=True)

    shap.summary_plot(shap_values, sample, feature_names,
                      max_display=50, show=False)
    utils.saveFig(dir_path, "shap_summary.png", show=True)

    sub_dir = dir_path / "dependence-plots"
    for i in range(len(feature_names)):
        shap.dependence_plot(i, shap_values, sample, feature_names, show=False,
                             title=("Change in Predicted Bike Sharing Demand "
                                    f"by Change in {feature_names[i]}"), )
        utils.saveFig(sub_dir, f"{feature_names[i]}.png")
