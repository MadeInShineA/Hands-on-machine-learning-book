from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans


def load_housing_data():
    data_file_path = Path("datasets/housing.tgz")
    if not data_file_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        data_url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
        urllib.request.urlretrieve(data_url, data_file_path)
        with tarfile.open(data_file_path) as housing_tgz:
            housing_tgz.extractall(path="datasets")
    return pd.read_csv("datasets/housing.csv")


def shuffle_and_split_data(data, test_ratio):
    indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def stratified_split_test_train(n_splits, test_size, data, stratified_by):
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    stratified_splits = []
    for train_index, test_index in splitter.split(data, data[stratified_by]):
        stratified_splits.append((data.loc[train_index], data.loc[test_index]))
    return stratified_splits

# Defining a custom transformer that computes Kmean clustering inside its fit method and a rbf_kernel inside its transform method


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=0.1, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(X, sample_weight=sample_weight)
        return self

    def transform(self ,X):
        return rbf_kernel(X, self.kmeans.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


if __name__ == '__main__':
    housing_data = load_housing_data()

    print("*** Dataset Informations ***\n")
    print(housing_data.info())

    print("\n*** Dataset Head***\n")

    print(housing_data.head())

    print("\n*** Ocean Proximity Value Counts ***\n")

    print(housing_data["ocean_proximity"].value_counts())

    print("\n*** Dataset Description ***\n")

    print(housing_data.describe())

    print("\n*** Dataset Histogram ***\n")

    housing_data.hist(bins=50, figsize=(20, 15))
    plt.show()

    # Creating the income_cat attribute to allow stratified sampling over the different incomes
    housing_data["income_cat"] = pd.cut(housing_data["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
    housing_data["income_cat"].hist()
    plt.show()

    # Splitting the data randomly
    train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)

    # Splitting the data using stratified sampling with the income categories we created
    stratified_train_set, stratified_test_set = stratified_split_test_train(1, 0.2, housing_data, "income_cat")[0]

    # Removing the income_cat attribute from the data
    for set_ in (stratified_train_set, stratified_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    original_stratified_train_set = stratified_train_set.copy()

    # Visualizing the data with the longitude and latitude attributes
    housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2,)
    plt.show()

    # Visualizing the data with the longitude and latitude attributes and the housing prices
    housing_data.plot(kind="scatter", x="longitude", y="latitude", grid=True, s=housing_data["population"]/100,
                      label="population", figsize=(10, 7), c="median_house_value", cmap=plt.get_cmap("jet"),
                      colorbar=True, legend=True, sharex=False)
    plt.show()

    # Looking for correlations
    print("*** Looking for correlations ***\n")

    print("\n* Default correlation matrix for the median house attribute *\n")
    correlation_matrix = housing_data.corr(numeric_only=True)
    print(correlation_matrix["median_house_value"].sort_values(ascending=False))

    correlation_attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing_data[correlation_attributes], figsize=(12, 8))
    plt.show()

    # The most promising attribute to predict the median house value seems to be the median income
    # Let's make its plot bigger

    housing_data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)
    plt.show()

    # Creating attributes to gain more insight on the data
    housing_data["rooms_per_house"] = housing_data["total_rooms"] / housing_data["households"]
    housing_data["bedrooms_ratio"] = housing_data["total_bedrooms"] / housing_data["total_rooms"]
    housing_data["people_per_house"] = housing_data["population"] / housing_data["households"]

    print("\n*** Correlation matrix for the median house value with newly created attributes ***\n")
    correlation_matrix = housing_data.corr(numeric_only=True)
    print(correlation_matrix["median_house_value"].sort_values(ascending=False))

    housing_data = stratified_train_set.drop("median_house_value", axis=1)
    housing_labels = stratified_train_set["median_house_value"].copy()

    # Modify the dataset to include the median of total_bedrooms when the attribute is empty
    median = housing_data["total_bedrooms"].median()
    housing_data["total_bedrooms"].fillna(median, inplace=True)

    # Using the SimpleImputer to fill the missing values inside the dataset
    imputer = SimpleImputer(strategy="median")

    # Retrieving only the numerical attributes of the dataset
    housing_num = housing_data.select_dtypes(include=[np.number])
    imputer.fit(housing_num)

    # Transforming the ocean_proximity attribute to a numerical attribute using OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_data[["ocean_proximity"]])

    # Transforming the ocean_proximity attribute to a numerical attribute using OneHotEncoder
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.handle_unknown = 'ignore'
    housing_cat_1hot = one_hot_encoder.fit_transform(housing_data[["ocean_proximity"]])

    # Transforming the numerical attributes to a range between 0 and 1 using MinMaxScaler
    min_max_scaler = MinMaxScaler()
    housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

    # Transforming the numerical attributes to a standard normal distribution using StandardScaler
    standard_scaler = StandardScaler()
    housing_num_standard_scaled = standard_scaler.fit_transform(housing_num)

    age_simil_35 = rbf_kernel(housing_data["housing_median_age"].values.reshape(-1, 1), [[35]], gamma=0.1)

    # Creating a custom transformer to compute the log values of the numerical attributes
    log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
    log_population = log_transformer.fit_transform(housing_data["population"])







    

