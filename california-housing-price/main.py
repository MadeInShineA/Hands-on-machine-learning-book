from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


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
    correlation_matrix = housing_data.corr()
    print(correlation_matrix["median_house_value"].sort_values(ascending=False))
