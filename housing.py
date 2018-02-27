import os
import shutil
import tarfile

import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_rooms_per_household=True, add_bedrooms_per_rooms=True, add_population_per_household=True):
        self.add_rooms_per_household = add_rooms_per_household
        self.add_bedrooms_per_rooms = add_bedrooms_per_rooms
        self.add_population_per_household = add_population_per_household

    def fit(self):
        return self

    def transform(self, housing):
        if self.add_rooms_per_household:
            housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
        if self.add_bedrooms_per_rooms:
            housing['bedrooms_per_rooms'] = housing['total_bedrooms'] / housing['total_rooms']
        if self.add_population_per_household:
            housing['population_per_household'] = housing['population'] / housing['households']
        return housing


def main():
    download_root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    housing_path = "datasets/housing"
    housing_url = download_root + housing_path + "/housing.tgz"
    fetch_housing_data(housing_url, housing_path)
    housing = load_housing_data(housing_path)
    strat_train_set, strat_test_set = split_train_test(housing, .2)
    housing = strat_train_set.drop('median_house_value', axis=1)
    housing_labels = strat_train_set['median_house_value'].copy()
    
    #housing = replace_missing_values(housing)
    #housing = encode_text_labels(housing)
    #transformer = CustomTransformer()
    #print(housing.head())
    #housing = transformer.transform(housing)
    #print(housing.head())


def fetch_housing_data(housing_url, housing_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    response = requests.get(housing_url, stream=True)
    response.raise_for_status()
    with open(tgz_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def split_train_test(housing, test_ratio):
    housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
    housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop(['income_cat'], axis=1, inplace=True)
    return strat_train_set, strat_test_set


def add_attributes(strat_train_set):
    housing = strat_train_set.copy()
    housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
    housing['bedrooms_per_rooms'] = housing['total_bedrooms'] / housing['total_rooms']
    housing['population_per_household'] = housing['population'] / housing['households']
    return housing


def visualize_geographical_data(housing):
    housing.plot(kind='scatter', x='longitude', y='latitude', alpha=.4, s=housing['population'] / 100,
                 label='population',
                 figsize=(10, 7), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
    plt.legend()
    plt.show()


def calc_corr_matrix(housing):
    corr_matrix = housing.corr()
    print(corr_matrix['median_house_value'].sort_values(ascending=False))


def plot_scatter_matrix(housing):
    attributes = ['median_house_value', 'median_income', 'total_rooms', 'bedrooms_per_rooms']
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()


def replace_missing_values(housing):
    imputer = Imputer(strategy='median')
    housing_num = housing.drop('ocean_proximity', axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr['ocean_proximity'] = housing['ocean_proximity']
    return housing_tr


def encode_text_labels(housing):
    encoder = LabelBinarizer()
    housing_cat = housing['ocean_proximity']
    housing_cat_hot = encoder.fit_transform(housing_cat)
    df = pd.DataFrame(housing_cat_hot, columns=encoder.classes_, index=housing.index)
    housing_tr = housing.drop('ocean_proximity', axis=1)
    housing_tr = pd.concat([housing_tr, df], axis=1)
    return housing_tr


if __name__ == '__main__':
    main()
