from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class NumericalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.original_features = X.columns
        self.scaler.fit(X, y)

        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def get_feature_names(self):
        return self.original_features


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.onehot = OneHotEncoder(handle_unknown="ignore")

    def fit(self, X, y=None):
        self.original_features = X.columns
        self.onehot.fit(X, y)

        return self.onehot

    def transform(self, X):
        return self.onehot.transform(X)

    def get_feature_names(self):
        return self.onehot.get_feature_names(self.original_features)


def getPreprocessingSteps(numerical_lst, categorical_lst):
    return ColumnTransformer(transformers=[
        ('num', NumericalEncoder(), numerical_lst),
        ('cat', CategoricalEncoder(), categorical_lst)
    ])
