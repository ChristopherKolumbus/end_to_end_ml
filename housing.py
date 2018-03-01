import os
import shutil
import tarfile

import requests
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from preprocessing import CategoricalEncoder


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attribute_names].values


rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_rooms_per_household=True, add_bedrooms_per_rooms=True, add_population_per_household=True):
        self.add_rooms_per_household = add_rooms_per_household
        self.add_bedrooms_per_rooms = add_bedrooms_per_rooms
        self.add_population_per_household = add_population_per_household

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.add_rooms_per_household:
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            X = np.c_[X, rooms_per_household]
        if self.add_bedrooms_per_rooms:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            X = np.c_[X, bedrooms_per_room]
        if self.add_population_per_household:
            population_per_household = X[:, population_ix] / X[:, households_ix]
            X = np.c_[X, population_per_household]
        return X



def main():
    # Download and load housing data:
    download_root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    housing_path = "datasets/housing"
    housing_url = download_root + housing_path + "/housing.tgz"
    fetch_housing_data(housing_url, housing_path)
    housing = load_housing_data(housing_path)
    # Prepare data:
    strat_train_set, strat_test_set = split_train_test(housing, .2)
    housing = strat_train_set.drop('median_house_value', axis=1)
    housing_labels = strat_train_set['median_house_value'].copy()
    num_attributes = list(housing)
    num_attributes.remove('ocean_proximity')
    cat_attributes = ['ocean_proximity']
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attributes)),
        ('imputer', Imputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attributes)),
        ('cat_encoder', CategoricalEncoder(encoding='onehot-dense'))
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline)
    ])
    model = Pipeline([
        ('feature_extraction', full_pipeline),
        ('regression', RandomForestRegressor())
    ])
    param_grid = [
        {
         'regression__bootstrap': [False],
         'regression__n_estimators': [100],
         'regression__max_features': [6]
         }
    ]
    grid_search = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error')
    grid_search.fit(housing, housing_labels)
    cv_results = grid_search.cv_results_
    for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
        print(f'{np.sqrt(-mean_score)}, {params}')
    feature_importances = grid_search.best_estimator_.named_steps['regression'].feature_importances_
    extra_attribs = ['rooms_per_household', 'bedrooms_per_room', 'population_per_room']
    cat_one_hot_attribs = list(grid_search.best_estimator_.named_steps['feature_extraction'].get_params()['cat_pipeline'].named_steps['cat_encoder'].categories_[0])
    attributes = num_attributes + extra_attribs + cat_one_hot_attribs
    print(sorted(zip(feature_importances, attributes), reverse=True))


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


def linear_regression(housing_prepared, housing_labels):
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(f'Linear regression RMSE: {lin_rmse}')


def tree_regression(housing_prepared, housing_labels):
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_predictions, housing_labels)
    tree_rmse = np.sqrt(tree_mse)
    print(f'Decision tree RMSE: {tree_rmse}')


def calc_model_rmse(housing_prepared, housing_labels, model='linear regression'):
    if model == 'linear regression':
        selected_model = LinearRegression()
    elif model == 'decision tree regression':
        selected_model = DecisionTreeRegressor()
    elif model == 'random forest regression':
        selected_model = RandomForestRegressor()
    selected_model.fit(housing_prepared, housing_labels)
    housing_predictions = selected_model.predict(housing_prepared)
    selected_model_mse = mean_squared_error(housing_predictions, housing_labels)
    selected_model_rmse = np.sqrt(selected_model_mse)
    print(f'{model} RMSE: {selected_model_rmse}')


def display_scores(scores):
    print(f'Scores: {scores}')
    print(f'Mean: {scores.mean()}')
    print(f'Standard deviation: {scores.std()}')


if __name__ == '__main__':
    main()
